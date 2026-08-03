import json
import torch
import random
import os
from api.models import Example
from storage import download_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from config import settings


"""
It needs one function: train_lora(job_id, dataset_s3_key, lora_config, template_version) -> dict
This function:

Downloads dataset from S3 using storage.download_dataset
Parses the JSONL into examples
Formats examples into prompt strings using template_version
Tokenizes and creates a HuggingFace dataset
Loads the base model and attaches LoRA adapters
Runs training with HuggingFace Trainer
Saves adapter to a local temp directory
Returns {"adapter_local_path": "...", "eval_loss": 1.517, "adapter_size_mb": 28.6}



"""


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(settings.base_model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class TextDataset(Dataset):
    def __init__(self, encodings, prompt_lengths=None):
        self.encodings = encodings
        self.prompt_lengths = prompt_lengths  # None = no masking (full-sequence loss)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]

        out = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["input_ids"], dtype=torch.long),
        }
        if self.prompt_lengths is not None:
            out["prompt_length"] = self.prompt_lengths[idx]
        return out


def collate_fn(batch):
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = input_ids.clone()

    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in loss

    # Prompt masking: if prompt_length is present, set labels[:prompt_length] = -100
    # so only response tokens contribute to the cross-entropy loss.
    if "prompt_length" in batch[0]:
        for i, b in enumerate(batch):
            prompt_len = b["prompt_length"]
            labels[i, :prompt_len] = -100  # mask prompt tokens

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


TEMPLATES = {
    "alpaca_v1": lambda text, label: (
        f"### Instruction:\n{text}\n\n### Response:\n{label}"
    )
}

# Prompt-only templates — same structure but WITHOUT the label.
# Used to compute prompt token length for prompt masking.
PROMPT_TEMPLATES = {
    "alpaca_v1": lambda text: f"### Instruction:\n{text}\n\n### Response:\n"
}


def format_examples(examples: list, template_version: str) -> list[str]:
    template_fn = TEMPLATES[template_version]
    return [template_fn(ex.text, ex.label) for ex in examples]


def compute_prompt_lengths(
    examples: list, template_version: str, max_length: int
) -> list[int]:
    """Tokenize prompt-only portions to find where the response starts.

    Returns a list of token counts — one per example. Labels tokens start
    at this index. Everything before it is prompt and should be masked (-100)
    when mask_prompt is enabled.
    """
    prompt_fn = PROMPT_TEMPLATES[template_version]
    lengths = []
    for ex in examples:
        prompt_only = prompt_fn(ex.text)
        prompt_ids = tokenizer(
            prompt_only, truncation=True, max_length=max_length, padding=False
        )
        lengths.append(len(prompt_ids["input_ids"]))
    return lengths


def parse_jsonl(content: bytes) -> list[Example]:
    lines = content.decode("utf-8").strip().split("\n")
    examples = []
    for line in lines:
        data = json.loads(line)
        examples.append(Example(text=data["text"], label=data["label"]))
    return examples


def split_dataset(examples, eval_ratio=0.1, seed=42):

    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    split = int(len(shuffled) * (1 - eval_ratio))

    return shuffled[:split], shuffled[split:]


num_workers = 0 if settings.device == "mps" else 2
LOG_EVERY = 1


def train_lora(
    job_id,
    dataset_s3_key,
    lora_config,
    template_version,
    redis_client,
    pg_conn,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    mask_prompt: bool = False,
) -> dict:
    # returns:
    # {
    # 'adapter_local_path': '/tmp/{job_id}/adapter'
    # 'eval_loss': 1.517,
    # 'adapter_size_mb': 28.6
    # ｝

    # Step 1 — download and parse
    content = download_dataset(dataset_s3_key)

    examples = parse_jsonl(content)  # create list of examples
    # examples = [Example(**d) for d in raw]

    train_examples, eval_examples = split_dataset(examples)  # split dataset

    # Step 2 — format and tokenize
    train_texts = format_examples(train_examples, template_version)
    eval_texts = format_examples(eval_examples, template_version)

    train_encodings = [
        tokenizer(text, truncation=True, max_length=max_length, padding=False)
        for text in train_texts
    ]
    eval_encodings = [
        tokenizer(text, truncation=True, max_length=max_length, padding=False)
        for text in eval_texts
    ]

    # Prompt masking: compute prompt token lengths so collate_fn can mask them
    train_prompt_lengths = None
    eval_prompt_lengths = None
    if mask_prompt:
        train_prompt_lengths = compute_prompt_lengths(
            train_examples, template_version, max_length
        )
        eval_prompt_lengths = compute_prompt_lengths(
            eval_examples, template_version, max_length
        )
    # Step 3 - Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        settings.base_model, dtype=settings.dtype, device_map=settings.device
    )

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=0.05,
        target_modules=lora_config.target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Step 4 - DataLoader

    train_dataset = TextDataset(train_encodings, prompt_lengths=train_prompt_lengths)
    eval_dataset = TextDataset(eval_encodings, prompt_lengths=eval_prompt_lengths)

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        shuffle=False,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Step 5 - Training loop

    optimizer = AdamW(
        model.parameters(),  # only adapter params have requires_grad=True
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            # move batch to device
            input_ids = batch["input_ids"].to(settings.device)
            attention_mask = batch["attention_mask"].to(settings.device)
            labels = batch["labels"].to(settings.device)

            # forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # backward pass
            loss.backward()

            # clip gradients - prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # update weights
            optimizer.step()

            # update lr
            scheduler.step()

            # Zero gradients - MUST happen before next forward pass
            optimizer.zero_grad()

            total_loss += loss.item()

            global_step = epoch * len(train_loader) + step

            # pubish to redis
            if global_step % LOG_EVERY == 0:
                redis_client.publish(
                    f"job:{job_id}:events",
                    json.dumps(
                        {
                            "type": "step",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "train_loss": round(loss.item(), 4),
                            "grad_norm": round(grad_norm.item(), 4),
                            "learning_rate": round(scheduler.get_last_lr()[0], 6),
                        }
                    ),
                )

                with pg_conn.cursor() as cur:
                    cur.execute(
                        """
                            INSERT INTO training_events (job_id, step, epoch, train_loss, grad_norm)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                        (job_id, global_step, epoch + 1, loss.item(), grad_norm.item()),
                    )
                pg_conn.commit()

            # CONTINUE

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | train_loss={avg_loss:.4f}")

        # Step 6 - Eval loop

        model.eval()

        total_eval_loss = 0.0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(settings.device)
                attention_mask = batch["attention_mask"].to(settings.device)
                labels = batch["labels"].to(settings.device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_eval_loss += loss.item()

        eval_loss = total_eval_loss / len(eval_loader)
        print(f"Epoch {epoch + 1} | eval_loss={eval_loss:.4f}")

        redis_client.publish(
            f"job:{job_id}:events",
            json.dumps(
                {"type": "eval", "epoch": epoch + 1, "eval_loss": round(eval_loss, 4)}
            ),
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                            INSERT INTO training_events (job_id, epoch, eval_loss)
                            VALUES (%s, %s, %s)
                            """,
                (job_id, epoch + 1, eval_loss),
            )

        pg_conn.commit()

        # CONTINUE

        # eval_loss = total_eval_loss / len(eval_loader)
        # print(f'Epoch {epoch+1} | eval_loss={eval_loss:.4f}')

        model.train()

    redis_client.publish(
        f"job:{job_id}:events",
        json.dumps(
            {"type": "done", "status": "complete", "eval_loss": round(eval_loss, 4)}
        ),
    )

    # Step 7 - Save and Return adapter
    # save adapter to temp directory
    tmp_dir = f"/tmp/{job_id}/adapter"
    os.makedirs(tmp_dir, exist_ok=True)
    model.save_pretrained(tmp_dir)  # saves ONLY adapter weights - not base

    tokenizer.save_pretrained(tmp_dir)  # saves tokenizer config

    # Compute adapter size
    total_bytes = sum(
        os.path.getsize(os.path.join(tmp_dir, f))
        for f in os.listdir(tmp_dir)
        if os.path.isfile(os.path.join(tmp_dir, f))
    )
    adapter_size_mb = total_bytes / 1e6

    return {
        "adapter_local_path": tmp_dir,
        "eval_loss": eval_loss,
        "adapter_size_mb": adapter_size_mb,
    }

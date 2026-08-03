from dataclasses import dataclass


@dataclass
class LoraConfig:
    r: int
    alpha: int
    target_modules: list[str]


ATTENTION_ONLY = ["q_proj", "v_proj"]
ATTENTION_FULL = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTENTION_AND_MLP = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def select_lora_config(num_examples: int) -> LoraConfig:

    if num_examples < 100:
        r = 8
        alpha = 16
        target_modules = ATTENTION_ONLY
        return LoraConfig(r, alpha, target_modules)

    elif num_examples >= 100 and num_examples < 500:
        r = 16
        alpha = 32
        target_modules = ATTENTION_FULL

        return LoraConfig(r, alpha, target_modules)

    elif num_examples >= 500 and num_examples < 2000:
        r = 32
        alpha = 64
        target_modules = ATTENTION_FULL
        return LoraConfig(r, alpha, target_modules)

    elif num_examples >= 2000:
        r = 64
        alpha = 128
        target_modules = ATTENTION_AND_MLP

        return LoraConfig(r, alpha, target_modules)

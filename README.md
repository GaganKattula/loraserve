# LoRA Serve — Design Package

LoRA Serve is a multi-tenant LoRA fine-tuning and inference service: users submit labeled datasets via a REST API, a decoupled GPU worker trains a LoRA adapter on a shared frozen base model and persists it to S3, and a serving engine hot-loads multiple adapters onto one GPU with an LRU cache so inference can switch between fine-tuned models without reloading from scratch. It's built as a two-machine system (always-on API/queue/database on a cheap VPS, on-demand GPU pod for training and serving) specifically to force real engagement with async job orchestration, live training telemetry over SSE, and PEFT-based multi-tenant model serving — problems that don't surface from training scripts alone.  

This folder contains the complete design documentation for LoRA Serve, a multi-tenant LoRA fine-tuning and inference service. It is intended for design review by Claude (large model) or human reviewers before implementation begins.

---

## Start Here

**Read `docs/PROJECT-INTRO.docx` first.** It explains what the project is, why it exists, how the documents relate to each other, and what questions remain open for review.

---

## Folder Structure

```
lora-serve-design/
├── docs/
│   ├── PROJECT-INTRO.docx    ← Start here. Project overview + navigation guide.
│   ├── HLD-v2.docx           ← High Level Design. Architecture, components, lifecycles.
│   └── LLD-v2.docx           ← Low Level Design. Schema, API, LRU, semaphore, watchdog.
│
└── diagrams/
    ├── HLD-v2.mermaid                ← Full system architecture flowchart
    ├── LLD-v2-training.mermaid       ← Training job sequence diagram
    ├── LLD-v2-inference.mermaid      ← Inference + LRU cache flowchart
    ├── LLD-v2-sse-semaphore.mermaid  ← SSE stream + GPU semaphore state machine
    └── LLD-v2-datamodel.mermaid      ← Postgres ER diagram
```

---

## Rendering Diagrams

Paste any `.mermaid` file content into [mermaid.live](https://mermaid.live) to render it visually. Alternatively, use Obsidian with the Mermaid plugin, or VS Code with the Markdown Preview Mermaid Support extension.

---

## What the Reviewer Should Do

1. Read `PROJECT-INTRO.docx` in full.
2. Review `HLD-v2.docx` and verify the two-machine architecture is sound.
3. Review `LLD-v2.docx` and answer the 7 open design questions in Section 8.
4. Check each Mermaid diagram for logical correctness against its corresponding docx section.
5. Flag any additional issues not captured in the existing open questions.

---

## Known Open Questions (Summary)

Full detail in `LLD-v2.docx` Section 8 and `PROJECT-INTRO.docx` Section 5.

| # | Question |
|---|---|
| Q1 | PeftModel (separate weights) vs merge_and_unload() (fused) for inference |
| Q2 | Second base model copy for concurrent training + inference |
| Q3 | Full dataset download before training vs streaming from S3 |
| Q4 | Size-based LRU eviction — is there a better strategy? |
| Q5 | Pre-warming N most recently trained adapters on serving engine startup |
| Q6 | GPU pod hostname registration — Postgres config table vs Redis key |
| Q7 | upload_pending retry when pod restarts and local adapter files are gone |

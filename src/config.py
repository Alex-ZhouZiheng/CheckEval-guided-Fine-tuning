"""
Shared configuration for the baseline experiments.

All paths, dataset settings, model defaults, and inference parameters live
here so every script in this folder stays aligned.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ────────────────────────── env ──────────────────────────────
# Walk up to find .env at the project root (thesis/)
_PROJECT_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_PROJECT_ROOT_ENV, override=False)

HF_TOKEN = os.getenv("HF_TOKEN")

# ────────────────────────── paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
WITH_REASON_DIR = DATA_DIR / "with_reason"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKLISTS_DIR = PROJECT_ROOT / "checklists" / "v4_frozen"

for _dir in [DATA_DIR, RAW_DIR, SPLITS_DIR, WITH_REASON_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────── dataset ──────────────────────────
HF_DATASET_ID = "nvidia/HelpSteer3"
KEEP_DOMAINS = {"general", "stem", "code"}
KEEP_PREFERENCES = {-3, -2, 2, 3}

# ────────────────────────── splits ───────────────────────────
DEV_RATIO = 0.10
SEED = 42

TIER_SIZES = {
    "tier_5k": 5_000,
    "tier_10k": 10_000,
    "tier_20k": 20_000,
}

# ────────────────────────── model / inference ────────────────
JUDGE_MODEL_ID = PROJECT_ROOT/"models"/"Qwen3.5-9B"
# Smaller backbone for the checklist-generator model (two-model pipeline).
GENERATOR_MODEL_ID = PROJECT_ROOT/"models"/"Qwen3.5-4B"

VLLM_ENGINE_KWARGS = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 16384,
    "dtype": "auto",
    "trust_remote_code": True,
    "language_model_only": True,
    "max_num_seqs": 16,
}

# Qwen3.5 thinks by default. Disable thinking for concise judge outputs.
VLLM_CHAT_KWARGS = {
    "chat_template_kwargs": {"enable_thinking": False},
}

# ────────────────────────── llama.cpp backend ────────────────
# Primary inference backend. "llamacpp" hits a locally-running llama-server
# via its OpenAI-compatible HTTP API; "vllm" uses the in-process vLLM engine.
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "vllm")

LLAMACPP_SERVER_URL = os.environ.get("LLAMACPP_SERVER_URL", "http://127.0.0.1:8080/v1")
LLAMACPP_API_KEY = os.environ.get("LLAMACPP_API_KEY", "EMPTY")
LLAMACPP_QUANT = os.environ.get("LLAMACPP_QUANT", "Q4_K_M")

LLAMACPP_SERVER_KWARGS = {
    "ctx_size": 16384,       # matches VLLM_ENGINE_KWARGS["max_model_len"]
    "parallel": 16,          # matches VLLM_ENGINE_KWARGS["max_num_seqs"]
    "n_gpu_layers": -1,      # all layers on GPU
    "flash_attn": True,
    "cont_batching": True,
}

# Client-side fan-out concurrency when calling llama-server.
LLAMACPP_HTTP_CONCURRENCY = int(os.environ.get("LLAMACPP_HTTP_CONCURRENCY", "16"))

GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
GGUF_ADAPTER_DIR = GGUF_DIR / "adapters"
JUDGE_GGUF = GGUF_DIR / f"Qwen3.5-9B.{LLAMACPP_QUANT}.gguf"
GENERATOR_GGUF = GGUF_DIR / f"Qwen3.5-4B.{LLAMACPP_QUANT}.gguf"

GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
}

# ────────────────────────── evaluation ───────────────────────
TIE_THRESHOLD = 0

# ────────────────────────── training / DPO ────────────────────
DPO_DIR = DATA_DIR / "dpo"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
TENSORBOARD_DIR = RESULTS_DIR / "tb_logs"

for _dir in [DPO_DIR, CHECKPOINTS_DIR,TENSORBOARD_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# Training hyperparameters
LEARNING_RATE = 1e-6
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
WARMUP_RATIO = 0.05
MAX_LENGTH = 2048


# DPO
DPO_BETA = 0.1

# ────────────────────────── joint training (DPO + checklist SFT) ──
CHECKLIST_SFT_DIR = DATA_DIR / "checklist_sft"
CHECKLIST_SFT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────── two-model pipeline ────────────────────
# SFT data for the separate checklist-generator and checklist-conditioned judge.
GENERATOR_SFT_DIR = DATA_DIR / "generator_sft"
JUDGE_SFT_DIR = DATA_DIR / "judge_sft"
# Generator inference output: one checklist per sample_id for each split.
GENERATED_CHECKLIST_DIR = DATA_DIR / "generated_checklists"
for _dir in [GENERATOR_SFT_DIR, JUDGE_SFT_DIR, GENERATED_CHECKLIST_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

JOINT_LAMBDA = 0.1          # weight of checklist SFT loss
SFT_MAX_LENGTH = 2048       # max token length for SFT samples

# DeepSpeed ZeRO-2
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

# Logging
WANDB_PROJECT = "Thesis"

# ────────────────────────── dynamic checklist dimensions ────────────────────────────

CHECKLIST_DIMENSIONS = [
    "clarity_and_communication",
    "coding_communication_conditional",
    "correctness_and_completeness",
    "helpfulness_and_usefulness",
    "relevance_instruction_following",
]

CHECKLIST_DIMENSION_DESCRIPTIONS = {
    "clarity_and_communication": (
        "The response communicates its content clearly enough that the user can "
        "extract the answer, understand the reasoning, and act on the result. "
        "In v4, this includes conditional structural effectiveness plus tone, "
        "style, register, persona, role-play, and narrative continuity signals."
    ),
    "coding_communication_conditional": (
        "Code-specific quality criteria that apply only when the response "
        "contains code, commands, configuration, APIs, syntax, or implementation "
        "details. Evaluate code accuracy, runnable completeness, API validity, "
        "context integration, explainability, and readability. Use NA for this "
        "dimension when no code or implementation detail is present."
    ),
    "correctness_and_completeness": (
        "The response is factually accurate, logically sound, and sufficiently "
        "complete to answer the user's actual question. In v4, this also covers "
        "ambiguity handling, false-premise correction, temporal awareness for "
        "time-sensitive facts, and avoiding unsupported certainty."
    ),
    "helpfulness_and_usefulness": (
        "The response provides substantive value beyond basic correctness by "
        "offering tailored guidance, actionable specifics, useful depth, "
        "relevant trade-offs, and an appropriate next step without drifting into "
        "generic or tangential material."
    ),
    "relevance_instruction_following": (
        "The response follows explicit instructions, requested output type, "
        "scope, format, numeric constraints, and source-grounded transformations. "
        "Use this dimension for task adherence, concrete format or length "
        "requirements, and whether given content is actually used or transformed "
        "as requested."
    ),
}

# Backward-compatible names used by existing generator / extractor scripts.
DOMAINS = CHECKLIST_DIMENSIONS
DOMAIN_DESCRIPTIONS = CHECKLIST_DIMENSION_DESCRIPTIONS

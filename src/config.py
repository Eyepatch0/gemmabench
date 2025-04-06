import os
from dotenv import load_dotenv
import warnings

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Don't import task_utils directly here to avoid circular dependency


def _load_tasks():
    from src.utils.task_utils import load_tasks_from_yaml
    return load_tasks_from_yaml()


if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in .env file. Access to gated models will fail.")
    warnings.warn(
        "HF_TOKEN not found in .env file. Please create a .env file with your Hugging Face token.")

# --- Constants ---
# Load tasks lazily to avoid circular imports
LIGHTEVAL_SUPPORTED_TASKS_SET = None  # Will be loaded when needed

# Lighteval specific task list url (for reference or future dynamic checks)
LIGHTEVAL_TASKS_URL = "https://raw.githubusercontent.com/huggingface/lighteval/refs/heads/main/docs/source/available-tasks.mdx"

LIGHTEVAL_BACKENDS = {
    "accelerate": "accelerate",
    "vllm": "vllm",
    "nanotron": "nanotron",
}

VALID_DTYPES = [
    "bfloat16",
    "float16",
    "float32",
    "auto",
]

RESULTS_DIR = "results"

# For now, only lighteval is supported
SUPPORTED_FRAMEWORK = "lighteval"

# Function to lazily load tasks when actually needed


def get_supported_tasks():
    global LIGHTEVAL_SUPPORTED_TASKS_SET
    if LIGHTEVAL_SUPPORTED_TASKS_SET is None:
        LIGHTEVAL_SUPPORTED_TASKS_SET = _load_tasks()
        if not LIGHTEVAL_SUPPORTED_TASKS_SET:
            print(
                "CRITICAL WARNING: No supported tasks loaded. Task validation will not work.")
    return LIGHTEVAL_SUPPORTED_TASKS_SET

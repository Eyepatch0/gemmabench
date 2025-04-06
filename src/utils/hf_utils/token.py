import logging
from huggingface_hub import HfFolder
from ...config import HF_TOKEN

logger = logging.getLogger(__name__)


def save_hf_token_globally() -> bool:
    if HF_TOKEN:
        try:
            HfFolder.save_token(HF_TOKEN)
            print("HF token saved globally for CLI tools.")
            return True
        except Exception as e:
            logger.error(f"Error saving HF token globally: {e}")
            print(f"Error: Could not save token globally: {e}")
            return False
    else:
        print("Warning: Cannot save token globally as HF_TOKEN is not set.")
        return False

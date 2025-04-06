import logging
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from ...config import HF_TOKEN

logger = logging.getLogger(__name__)


def check_model_exists(model_id: str) -> bool:
    print(f"Verifying model '{model_id}' on Hugging Face Hub...")
    api = HfApi()
    try:
        api.model_info(model_id, token=HF_TOKEN)
        print(f"Model '{model_id}' found.")
        return True
    except RepositoryNotFoundError:
        print(f"Error: Model '{model_id}' not found on Hugging Face Hub.")
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while checking the model: {e}")
        print(f"An unexpected error occurred while checking the model: {e}")
        return False

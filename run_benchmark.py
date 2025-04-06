import sys
import os
from src.config import HF_TOKEN, SUPPORTED_FRAMEWORK, LIGHTEVAL_BACKENDS, RESULTS_DIR, VALID_DTYPES
from src.utils.hf_utils import check_model_exists
from src.utils.system_utils import get_system_info, display_system_info, recommend_backend
from src.utils.task_utils import get_task_details_interactive
from src.frameworks import LightevalRunner  # Import specific runner for Demo
# from src.frameworks import LmEvalHarnessRunner # Future


def main():
    print("Welcome to Gemmabench!")

    framework = SUPPORTED_FRAMEWORK
    print(f"Using benchmark framework: {framework}")
    if framework == "lighteval":
        runner_class = LightevalRunner
    # elif framework == "lm-eval-harness":
    #     runner_class = LmEvalHarnessRunner
    else:
        print(f"Error: Framework '{framework}' is not supported.")
        sys.exit(1)

    # --- Hugging Face Token Check ---
    if not HF_TOKEN:
        print("\nError: Hugging Face token (HF_TOKEN) not found in your environment.")
        print("Please create a .env file in the root directory with your token:")
        print("HF_TOKEN=your_actual_token")
        print("You can get a token from https://huggingface.co/settings/tokens")
        print("Benchmarking gated models like Gemma will likely fail without it.")
        proceed = input("Proceed anyway? (y/N): ").lower()
        if proceed != 'y':
            sys.exit(1)

    while True:
        model_id = input(
            "Enter the Hugging Face model ID (e.g., google/gemma-7b): ").strip()
        if not model_id:
            print("Model ID cannot be empty.")
            continue
        if check_model_exists(model_id):
            break
        else:
            print("Please try entering the model ID again.")

    display_sys_info = input(
        "Check system resources (CPU/RAM/GPU)? (y/N): ").lower()
    selected_backend = None
    if display_sys_info == 'y':
        print("Gathering system info...")
        system_info = get_system_info()
        display_system_info(system_info)
        recommended_backend = recommend_backend(system_info)
        print(
            f"Recommended backend based on system info: '{recommended_backend}'")

        available_backends = list(LIGHTEVAL_BACKENDS.keys())
        print(f"Available backends for {framework}: {available_backends}")
        while True:
            backend_input = input(
                f"Choose a backend or press Enter to use recommendation ('{recommended_backend}'): ").strip().lower()
            if not backend_input:
                selected_backend = recommended_backend
                break
            elif backend_input in available_backends:
                selected_backend = backend_input
                break
            else:
                print(
                    f"Invalid backend. Please choose from: {available_backends}")
    else:
        available_backends = list(LIGHTEVAL_BACKENDS.keys())
        print(f"\nAvailable backends for {framework}: {available_backends}")
        while True:
            default_backend = "accelerate"
            backend_input = input(
                f"Choose a backend [default: {default_backend}]: ").strip().lower()
            if not backend_input:
                selected_backend = default_backend
                break
            elif backend_input in available_backends:
                selected_backend = backend_input
                break
            else:
                print(
                    f"Invalid backend. Please choose from: {available_backends}")

    print(f"Selected backend: {selected_backend}")

    dtype = "auto"
    if selected_backend == "vllm":
        print(f"\nAvailable dtypes for vLLM: {VALID_DTYPES}")
        while True:
            dtype_input = input(
                f"Choose dtype for vLLM [default: auto]: ").strip().lower()
            if not dtype_input:
                dtype = "auto"
                break
            elif dtype_input in VALID_DTYPES:
                dtype = dtype_input
                break
            else:
                print(f"Invalid dtype. Please choose from: {VALID_DTYPES}")
        print(f"Selected dtype: {dtype}")

    task_details = get_task_details_interactive()
    if not task_details:
        print("Failed to get valid task details. Exiting.")
        sys.exit(1)

    runner = runner_class(model_id=model_id, hf_token=HF_TOKEN)

    kwargs = {}
    if selected_backend == "vllm":
        kwargs["dtype"] = dtype

    success = runner.run(task_details=task_details,
                         backend=selected_backend, **kwargs)

    if success:
        print("\nBenchmark process completed.")
        print(
            f"Check the '{RESULTS_DIR}/{runner.framework_name()}/' directory for results.")
    else:
        print("\nBenchmark process failed.")
        sys.exit(1)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gitkeep_path = os.path.join(RESULTS_DIR, ".gitkeep")
    if not os.path.exists(gitkeep_path):
        try:
            with open(gitkeep_path, 'w') as f:
                pass
        except IOError:
            print(f"Warning: Could not create .gitkeep in {RESULTS_DIR}")

    main()

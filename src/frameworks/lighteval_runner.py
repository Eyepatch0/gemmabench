import subprocess
import os
import json
import datetime
import shlex
from typing import Dict
from ..benchmarker import BenchmarkRunner
from ..config import LIGHTEVAL_BACKENDS, VALID_DTYPES


class LightevalRunner(BenchmarkRunner):
    @staticmethod
    def framework_name() -> str:
        return "lighteval"

    def run(self, task_details: Dict, backend: str, **kwargs) -> bool:
        print(f"\nStarting lighteval benchmark for model: {self.model_id}")
        print(f"Task details: {task_details}")
        print(f"Using backend: {backend}")

        if backend not in LIGHTEVAL_BACKENDS:
            print(
                f"Error: Backend '{backend}' is not recognized for lighteval.")
            print(f"Supported backends: {list(LIGHTEVAL_BACKENDS.keys())}")
            return False

        lighteval_launcher = LIGHTEVAL_BACKENDS[backend]

        task_identifier = task_details['task_identifier']
        num_few_shot = task_details['num_few_shot']
        allow_truncation = task_details['allow_truncation']  # Boolean
        allow_truncation_int = int(allow_truncation)

        command = ["lighteval", lighteval_launcher]

        # --- Model arguments string ---
        model_args_list = [
            f"pretrained={self.model_id}",
            "trust_remote_code=True"
        ]

        if backend == "vllm":
            model_args_str = ",".join(model_args_list)
            if "dtype=" not in model_args_str:
                selected_dtype = kwargs.get("dtype", "auto")
                if selected_dtype not in VALID_DTYPES:
                    print(
                        f"Warning: Invalid dtype '{selected_dtype}'. Using 'auto' instead.")
                    selected_dtype = "auto"

                print(f"Using vLLM backend with dtype={selected_dtype}")
                model_args_list.append(f"dtype={selected_dtype}")
            else:
                print(
                    "Warning: dtype already specified in model_args_list. Skipping automatic addition.")

        model_args_string = ",".join(model_args_list)
        command.append(model_args_string)

        # --- Task arguments string ---
        # Format: suite|task_name:num_few_shot:allow_truncation
        task_string = f"{task_identifier}|{num_few_shot}|{allow_truncation_int}"
        command.append(task_string)

        # --- Optional arguments ---
        # Override batch size (optional)
        if backend != "vllm":
            batch_size = kwargs.get("override-batch-size", 1)
            command.extend(["--override-batch-size", str(batch_size)])

        # Output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = self.model_id.replace("/", "_")
        safe_task_string = task_string.replace(":", "_")
        run_output_dir_name = f"{safe_model_name}_{safe_task_string}_{backend}_{timestamp}"
        run_output_dir = os.path.join(self.results_dir, run_output_dir_name)
        command.extend(["--output-dir", run_output_dir])

        print("\nExecuting command:")
        print(shlex.join(command))

        try:
            process = subprocess.run(
                command, capture_output=True, text=True, check=True, env=os.environ)
            print("\n--- lighteval stdout ---")
            print(process.stdout)
            if process.stderr:
                print("--- lighteval stderr ---")
                print(process.stderr)
            print("\nBenchmark finished successfully.")

            results_file = os.path.join(run_output_dir, "results.json")
            if os.path.exists(results_file):
                print(f"Results saved in directory: {run_output_dir}")
                print(f"Main results file: {results_file}")
            else:
                print(
                    f"Benchmark completed, but results.json not found in {run_output_dir}. Check logs/stdout.")

            return True

        except FileNotFoundError:
            print("\nError: 'lighteval' command not found.")
            print(
                "Please ensure lighteval is installed correctly in your environment (pip install lighteval...).")
            return False
        except subprocess.CalledProcessError as e:
            print("\nError: lighteval command failed.")
            print(f"Return code: {e.returncode}")
            print("--- stdout ---")
            print(e.stdout)
            print("--- stderr ---")
            print(e.stderr)
            return False
        except Exception as e:
            print(
                f"\nAn unexpected error occurred during benchmark execution: {e}")
            return False

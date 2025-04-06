from typing import Optional, Dict, Any
from .task_validator import validate_task
from .task_discovery import get_available_task_suites
from ...config import get_supported_tasks


def get_task_details_interactive() -> Optional[Dict[str, Any]]:
    available_suites = get_available_task_suites()

    print("\n--- Task Selection ---")
    print(f"Available task suites: {', '.join(available_suites)}")

    while True:
        task_suite = input(
            "Enter the task suite you want to run (e.g., 'mmlu', 'bigbench'): ").strip().lower()
        if not task_suite:
            print("Task suite cannot be empty.")
            continue

        if task_suite not in available_suites:
            print(
                f"Error: '{task_suite}' is not a valid task suite. Please choose from: {', '.join(available_suites)}")
            continue
        break

    suite_tasks = [
        t for t in get_supported_tasks() if t.startswith(f"{task_suite}|")]

    print(f"\nAvailable tasks for suite '{task_suite}':")
    # Show first 20 tasks to avoid overwhelming output
    for i, task in enumerate(suite_tasks[:20]):
        print(f"  - {task}")
    if len(suite_tasks) > 20:
        print(f"  ... and {len(suite_tasks)-20} more tasks.")

    while True:
        task_input = input(
            f"Enter the full task identifier (e.g., '{suite_tasks[0] if suite_tasks else task_suite + '|example'}'): ").strip()
        if not task_input:
            print("Task identifier cannot be empty.")
            continue

        if not validate_task(task_input):
            retry = input(
                "Would you like to try a different task? (y/N): ").lower()
            if retry != 'y':
                return None
            continue

        print(f"Proceeding with task: {task_input}")
        break

    # Get number of few-shot examples
    num_few_shot = 5  # Default
    while True:
        try:
            num_few_shot_str = input(
                "Enter number of few-shot examples [default: 5]: ").strip()
            if not num_few_shot_str:
                break

            num_few_shot = int(num_few_shot_str)
            if num_few_shot < 0:
                print("Number of few-shot examples cannot be negative.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Get truncation setting
    allow_truncation = 1  # Default
    while True:
        allow_truncation_str = input(
            "Allow truncation if context is too small? (0 for No, 1 for Yes) [default: 1]: ").strip()
        if not allow_truncation_str:
            break

        try:
            allow_truncation = int(allow_truncation_str)
            if allow_truncation not in [0, 1]:
                print("Invalid input. Please enter 0 or 1.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter 0 or 1.")

    return {
        "task_identifier": task_input,
        "num_few_shot": num_few_shot,
        "allow_truncation": allow_truncation
    }

import warnings
import yaml
import os
from typing import Set, Dict, Any


def load_tasks_from_yaml(filepath=None) -> Set[str]:
    if filepath is None:
        # Default path relative to the project root
        filepath = os.path.join(
            "src", "data", "lighteval_supported_tasks.yaml")

    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        # Flatten the list of tasks from the structured YAML
        task_list = []
        if isinstance(data, dict) and "tasks" in data:
            for suite, tasks_in_suite in data["tasks"].items():
                if isinstance(tasks_in_suite, list):
                    task_list.extend(tasks_in_suite)
                else:
                    warnings.warn(
                        f"Expected a list under suite '{suite}' in {filepath}, but got {type(tasks_in_suite)}. Skipping.")
            return set(task_list)  # Return a set for fast lookups
        else:
            warnings.warn(
                f"Could not find top-level 'tasks' key or structure is incorrect in {filepath}. Returning empty set.")
            return set()
    except FileNotFoundError:
        warnings.warn(
            f"YAML task file not found at {filepath}. Task validation will fail.")
        return set()
    except yaml.YAMLError as e:
        warnings.warn(
            f"Error parsing YAML file {filepath}: {e}. Task validation will fail.")
        return set()
    except Exception as e:
        warnings.warn(
            f"An unexpected error occurred loading tasks from {filepath}: {e}")
        return set()

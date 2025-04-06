from ...config import LIGHTEVAL_TASKS_URL, get_supported_tasks


def validate_task(task_name: str) -> bool:
    supported_tasks = get_supported_tasks()

    if task_name in supported_tasks:
        print(f"Task '{task_name}' is supported.")
        return True

    print(f"Error: Task '{task_name}' is not supported.")
    print(f"For a full list of tasks in lighteval, see: {LIGHTEVAL_TASKS_URL}")
    return False

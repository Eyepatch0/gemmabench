from typing import List
from ...config import get_supported_tasks


def get_available_task_suites() -> List[str]:
    suites = set()
    for task in get_supported_tasks():
        if '|' in task:
            suite = task.split('|')[0]
            suites.add(suite)
    return sorted(list(suites))

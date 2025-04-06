import subprocess
import io
import csv
import warnings
from typing import Dict, List, Any, Optional


def _run_nvidia_smi() -> Optional[List[Dict[str, Any]]]:
    gpu_info_list = []
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding='utf-8')
        csvfile = io.StringIO(result.stdout)
        reader = csv.reader(csvfile)

        for row in reader:
            if len(row) == 5:
                try:
                    index = int(row[0].strip())
                    name = row[1].strip()
                    total_mem_mib = int(row[2].strip())
                    used_mem_mib = int(row[3].strip())
                    free_mem_mib = int(row[4].strip())

                    gpu_info_list.append({
                        "index": index,
                        "name": name,
                        "memory_total_gb": round(total_mem_mib / 1024, 2),
                        "memory_used_gb": round(used_mem_mib / 1024, 2),
                        "memory_free_gb": round(free_mem_mib / 1024, 2),
                    })
                except (ValueError, IndexError) as parse_err:
                    warnings.warn(
                        f"Could not parse nvidia-smi output row: {row}. Error: {parse_err}")
                    continue
        return gpu_info_list

    except FileNotFoundError:
        warnings.warn(
            "'nvidia-smi' command not found. Cannot get detailed GPU memory usage. Is the NVIDIA driver installed and nvidia-smi in PATH?")
        return None
    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"nvidia-smi command failed with error code {e.returncode}: {e.stderr}")
        return None
    except Exception as e:
        warnings.warn(
            f"An unexpected error occurred while running nvidia-smi: {e}")
        return None

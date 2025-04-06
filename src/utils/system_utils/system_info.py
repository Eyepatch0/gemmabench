import psutil
import torch
import platform
import warnings
from typing import Dict, Any

from .nvidia import _run_nvidia_smi


def get_system_info() -> Dict[str, Any]:
    info = {}

    # --- CPU ---
    info['platform'] = platform.system()
    info['architecture'] = platform.machine()
    info['cpu_cores'] = psutil.cpu_count(logical=False)
    info['cpu_logical_processors'] = psutil.cpu_count(logical=True)

    # --- RAM ---
    svmem = psutil.virtual_memory()
    info['ram_total_gb'] = round(svmem.total / (1024**3), 2)
    info['ram_available_gb'] = round(svmem.available / (1024**3), 2)

    # --- GPU ---
    info['gpu_available_torch'] = torch.cuda.is_available()
    info['gpu_devices'] = []
    info['gpu_driver_info'] = None

    # Prioritize nvidia-smi if platform is Linux/Windows and NVIDIA GPU expected
    if info['platform'] in ["Linux", "Windows"]:
        nvidia_smi_data = _run_nvidia_smi()
        if nvidia_smi_data is not None:
            info['gpu_available'] = len(nvidia_smi_data) > 0
            info['gpu_count'] = len(nvidia_smi_data)
            info['gpu_devices'] = nvidia_smi_data
            info['gpu_source'] = "nvidia-smi"

            if info['gpu_available'] and not info['gpu_available_torch']:
                warnings.warn(
                    "nvidia-smi found GPUs, but torch.cuda.is_available() is False. Check PyTorch installation with CUDA support.")
            elif not info['gpu_available'] and info['gpu_available_torch']:
                warnings.warn(
                    "torch.cuda.is_available() is True, but nvidia-smi failed or found no GPUs. Check driver/runtime consistency.")
        else:
            # Fallback to PyTorch method if nvidia-smi failed but torch detects CUDA
            info['gpu_source'] = "pytorch (nvidia-smi failed)"
            info['gpu_available'] = info['gpu_available_torch']

            if info['gpu_available']:
                try:
                    num_gpus = torch.cuda.device_count()
                    info['gpu_count'] = num_gpus

                    for i in range(num_gpus):
                        gpu_torch_info = {
                            'index': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_total_gb': round(
                                torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                            'memory_free_gb': "N/A (use nvidia-smi)"
                        }
                        info['gpu_devices'].append(gpu_torch_info)
                except Exception as e:
                    info['gpu_available'] = False
                    info['gpu_error'] = f"PyTorch GPU info error: {e}"
                    warnings.warn(
                        f"Could not get detailed GPU info via PyTorch: {e}")
            else:
                info['gpu_available'] = False
                info['gpu_count'] = 0
    else:
        info['gpu_source'] = "pytorch"
        # On macOS this checks for 'mps'
        info['gpu_available'] = info['gpu_available_torch']

        if info['gpu_available']:
            if torch.backends.mps.is_available():  # Check specifically for Apple Silicon MPS
                info['gpu_count'] = 1  # Typically only one MPS device
                info['gpu_devices'].append({
                    "index": 0,
                    "name": "Apple MPS",
                    "memory_total_gb": "N/A",  # Total/Free memory not easily queryable for MPS via torch
                    "memory_free_gb": "N/A",
                })
            else:
                info['gpu_available'] = False
                info['gpu_count'] = 0
                warnings.warn(
                    "torch.cuda.is_available() is True on non-Linux/Windows, but MPS not found. Unexpected state.")
        else:
            info['gpu_available'] = False
            info['gpu_count'] = 0

    return info


def display_system_info(info: Dict[str, Any]) -> None:
    print("\n--- System Information ---")
    print(
        f"Platform: {info.get('platform', 'N/A')} ({info.get('architecture', 'N/A')})")
    print(
        f"CPU Cores: {info.get('cpu_cores', 'N/A')} (Physical), {info.get('cpu_logical_processors', 'N/A')} (Logical)")
    print(f"RAM: {info.get('ram_available_gb', 'N/A')} GB Available / {info.get('ram_total_gb', 'N/A')} GB Total")

    print(f"GPU Source: {info.get('gpu_source', 'N/A')}")
    if info.get('gpu_available', False):
        print(f"GPU: Detected {info.get('gpu_count', 0)} device(s)")
        for i, gpu in enumerate(info.get('gpu_devices', [])):
            index = gpu.get('index', i)
            name = gpu.get('name', 'N/A')
            total_mem = gpu.get('memory_total_gb', 'N/A')
            free_mem = gpu.get('memory_free_gb', 'N/A')
            used_mem = gpu.get('memory_used_gb', 'N/A')
            print(
                f"  GPU {index}: {name} | Total Mem: {total_mem} GB | Free Mem: {free_mem} GB | Used Mem: {used_mem} GB")
    elif 'gpu_error' in info:
        print(f"GPU: Error detecting GPU details: {info['gpu_error']}")
    else:
        torch_cuda_status = "available" if info.get(
            'gpu_available_torch') else "not available"
        print(
            f"GPU: No compatible devices detected by '{info.get('gpu_source', 'N/A')}'. (PyTorch CUDA status: {torch_cuda_status})")
    print("--------------------------\n")

    # Issue hardware warnings if needed
    if isinstance(info.get('ram_available_gb'), (int, float)) and info['ram_available_gb'] < 16:
        print(
            "⚠️  NOTE: Available RAM is less than 16GB. Large models might lead to performance issues or errors.")
    if info.get('gpu_source') == 'nvidia-smi' and info.get('gpu_devices'):
        first_gpu_free_mem_gb = info['gpu_devices'][0].get('memory_free_gb')
        low_gpu_mem_threshold = 8
        if isinstance(first_gpu_free_mem_gb, (int, float)) and first_gpu_free_mem_gb < low_gpu_mem_threshold:
            print(
                f"⚠️  NOTE: Available GPU memory on device 0 is less than {low_gpu_mem_threshold}GB ({first_gpu_free_mem_gb}GB free). Loading large models might fail.")

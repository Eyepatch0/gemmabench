from typing import Dict, Any


def recommend_backend(system_info: Dict[str, Any]) -> str:
    if system_info.get('gpu_available', False) and system_info.get('gpu_devices'):
        # Use total memory of the first GPU for the heuristic
        first_gpu = system_info['gpu_devices'][0]
        first_gpu_mem_gb = first_gpu.get('memory_total_gb', 0)

        if isinstance(first_gpu_mem_gb, (int, float)):
            if first_gpu_mem_gb >= 20:
                print("Recommendation: Sufficient total GPU memory detected.")
                recommended = "vllm"  # VLLM is often faster for large batches/throughput if memory allows
            elif first_gpu_mem_gb >= 10:
                print("Recommendation: Moderate total GPU memory detected.")
                recommended = "accelerate"
            else:
                print("Recommendation: Limited total GPU memory detected.")
                recommended = "accelerate"  # Accelerate might handle lower memory better
        else:
            print("Recommendation: Could not determine numeric GPU memory.")
            recommended = "accelerate"

        # Show free memory if available
        if system_info.get('gpu_source') == 'nvidia-smi':
            first_gpu_free_mem = first_gpu.get('memory_free_gb', 'N/A')
            print(
                f"  -> Note: Currently free memory on GPU 0 is approx. {first_gpu_free_mem} GB.")

        return recommended
    else:
        print("Recommendation: No compatible GPU detected or issue obtaining GPU info.")
        return "accelerate"

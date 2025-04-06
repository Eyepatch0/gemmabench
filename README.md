# GemmaBench

GemmaBench is a tool for benchmarking [Gemma](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b) and other Hugging Face language models using standard benchmarking tasks.

## Features

- Benchmark Gemma and other Hugging Face models
- Integration with LightEval benchmarking framework
- Multiple backend options (Accelerate, vLLM, Nanotron)
- System resource detection and backend recommendations
- Supports all benchmarks available in `lighteval`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemmabench.git
cd gemmabench

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your Hugging Face token:
```bash
HF_TOKEN=your_hugging_face_token
```

## Usage

To benchmark a model, run the following command:
```bash
python run_benchmark.py
```
The script will guide you through:

- Selecting a model
- Choosing a backend (accelerate, vllm, nanotron)
- Configuring task parameters
- Running the benchmark

## Backends

- accelerate: Default backend, works on most systems
- vllm: Better performance for models that fit in GPU memory
- nanotron: Specialized for certain model architectures

## Supported Tasks
Currently, the tool supports the MMLU benchmark suite (Multi-task Language Understanding).

## System Requirements
- Python 3.8+
- For GPU acceleration:
    - NVIDIA GPU with appropriate drivers
    - CUDA toolkit compatible with your PyTorch installation

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## TODOs:
- [] Add support for benchmarking local models instead of just Hugging Face models
- [] Add support for running benchmarks with `lm-evaluation-harness`
- [] Improve recommendations for backend selection instead of the current approach.
- [] Add more options for customizing benchmark selection
- [] Add visualization of benchmark results, and live updates during benchmarking
- [] Add better error handling and user feedback
- [] Add feature for comparing multiple models at once
- [] Add an agent to consider natural language instructions for benchmarking


## Demo:
## Demo

Here's an example of running `gemmabench` to evaluate the `google/gemma-3-1b-it` model on the `helm|mmlu:anatomy` task using the `accelerate` backend:

```bash
(venv) (base) root@syeds:~/projects/gemmabench# python run_benchmark.py
Welcome to Gemmabench!
Using benchmark framework: lighteval
Enter the Hugging Face model ID (e.g., google/gemma-7b): google/gemma-3-1b-it
Verifying model 'google/gemma-3-1b-it' on Hugging Face Hub...
Model 'google/gemma-3-1b-it' found.
Check system resources (CPU/RAM/GPU)? (y/N): y
Gathering system info...

--- System Information ---
Platform: Linux (x86_64)
CPU Cores: 4 (Physical), 8 (Logical)
RAM: 4.53 GB Available / 7.65 GB Total
GPU Source: nvidia-smi
GPU: Detected 1 device(s)
  GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU | Total Mem: 4.0 GB | Free Mem: 3.59 GB | Used Mem: 0.28 GB
--------------------------

⚠️  NOTE: Available RAM is less than 16GB. Large models might lead to performance issues or errors.
⚠️  NOTE: Available GPU memory on device 0 is less than 8GB (3.59GB free). Loading large models might fail.
Recommendation: Limited total GPU memory detected.
  -> Note: Currently free memory on GPU 0 is approx. 3.59 GB.
Recommended backend based on system info: 'accelerate'
Available backends for lighteval: ['accelerate', 'vllm', 'nanotron']
Choose a backend or press Enter to use recommendation ('accelerate'): accelerate
Selected backend: accelerate

--- Task Selection ---
Available task suites: bigbench, harness, helm, leaderboard, lighteval, original
Enter the task suite you want to run (e.g., 'mmlu', 'bigbench'): helm

Available tasks for suite 'helm':
  - helm|wikifact:movement
  - helm|mmlu:college_physics
  # ... (list truncated for brevity) ...
  - helm|numeracy:linear_standard
  ... and 436 more tasks.
Enter the full task identifier (e.g., 'helm|wikifact:movement'): helm|mmlu:anatomy
Task 'helm|mmlu:anatomy' is supported.
Proceeding with task: helm|mmlu:anatomy
Enter number of few-shot examples [default: 5]: 5
Allow truncation if context is too small? (0 for No, 1 for Yes) [default: 1]: 1

Starting lighteval benchmark for model: google/gemma-3-1b-it
Task details: {'task_identifier': 'helm|mmlu:anatomy', 'num_few_shot': 5, 'allow_truncation': 1}
Using backend: accelerate

Executing command:
lighteval accelerate pretrained=google/gemma-3-1b-it,trust_remote_code=True 'helm|mmlu:anatomy|5|1' --override-batch-size 1 --output-dir 'results/lighteval/google_gemma-3-1b-it_helm|mmlu_anatomy|5|1_accelerate_20250405_194950'

--- lighteval stdout ---
INFO 04-05 19:49:58 [__init__.py:239] Automatically detected platform cuda.
# ... (lighteval logs and progress bar omitted for brevity) ...
Greedy generation: 100%|██████████| 135/135 [00:32<00:00,  4.77it/s]
Splits: 100%|██████████| 1/1 [00:32<00:00, 32.10s/it]
|       Task        |Version|Metric|Value |   |Stderr|
|-------------------|------:|------|-----:|---|-----:|
|all                |       |em    |0.4222|±  |0.0427|
|                   |       |qem   |0.4222|±  |0.0427|
|                   |       |pem   |0.4222|±  |0.0427|
|                   |       |pqem  |0.5333|±  |0.0431|
|helm:mmlu:anatomy:5|      0|em    |0.4222|±  |0.0427|
|                   |       |qem   |0.4222|±  |0.0427|
|                   |       |pem   |0.4222|±  |0.0427|
|                   |       |pqem  |0.5333|±  |0.0431|


--- lighteval stderr ---
# ... (detailed logs omitted for brevity) ...
[2025-04-05 19:50:46,815] [    INFO]: Saving results to /root/projects/gemmabench/results/lighteval/google_gemma-3-1b-it_helm|mmlu_anatomy|5|1_accelerate_20250405_194950/results/google/gemma-3-1b-it/results_2025-04-05T19-50-46.623637.json (evaluation_tracker.py:234)


Benchmark finished successfully.
(venv) (base) root@syeds:~/projects/gemmabench# 
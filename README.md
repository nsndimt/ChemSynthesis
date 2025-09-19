Code for "From Literature to Lab: LLM powered catalyst synthesis Protocol Generation"

Dataset are hosted at Zenodo [URL](https://zenodo.org/records/16948950)

Due to copyright issues, please email [zhangyue@udel.edu](mailto:zhangyue@udel.edu) or [hfang@udel.edu](mailto:hfang@udel.edu)  to get download access of the zenodo dataset.

# Environment Setup
First, load the required system modules and create a dedicated conda environment.
```shell
# Reset environment and load GPU stack
module reset
module load anaconda3_gpu
module load cuda/12.4.0
module list

# Setup conda
source /sw/external/python/anaconda3/etc/profile.d/conda.sh

# Create environment
conda create --name llamafactory python==3.10
conda activate llamafactory

# Install dependencies
pip install torch==2.5.1 transformers==4.49.0 accelerate==1.2.1 \
            datasets==3.2.0 peft==0.12.0 trl==0.9.6 deepspeed==0.16.2 \
            vllm==0.7.2

# Clone and install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt
pip install --no-deps -e .
```

# Dataset Preparation

1. Create a directory for checkpoints and datasets: ``mkdir checkpoint_llamafactory``
2. Download ``llamafactory_dataset.tar.gz`` from Zenodo Dataset and unzip this file
3. Place all your training/evaluation and dataset description JSON files inside ``checkpoint_llamafactory`` folder; 
4. Keep the Slurm and training scripts in the repo (``LLaMA-Factory``) root.:

```bash
stage1.slurm
stage1.sh
stage2.slurm
stage2.sh
checkpoint_llamafactory/
  ├── stage1_train.json
  ├── stage1_eval.json
  ├── stage2_train.json
  ├── stage2_eval.json
  ├── dataset_info.json
  ├── Llama-3.3-70b-instruct_stage2_pred.json
  ......
```

# Training/Testing Script

The main training pipeline is defined in ``stage1.sh`` and ``stage2.sh``. It performs:
- Model download from Hugging Face Hub
- LoRA fine-tuning with DeepSpeed
- Adapter export and LoRA merging
- Evaluation with vllm_infer.py
- Syncing checkpoints/logs/predictions back to ``$HOME/checkpoint_llamafactory``

## End-to-End Testing

Stage 2 cannot be directly tested using ``stage2_eval.json`` since it is human annotated, not the actuall prediction of LLM.

Users should collect ``pred.jsonl`` from Stage 1 LLM folders under ``$HOME/checkpoint_llamafactory``

To enable these prediction to be used by LLaMA-Factory, they must be registered in ``dataset_info.json`` by adding a entry of its name and path

```json
    "Llama-3.3-70b-instruct_stage2_pred": {
        "file_name": "Llama-3.3-70b-instruct_stage2_pred.json"
    }
```
### Reproduce Stage 2 Prediction
All stage 1 LLM predictions (e.g., ``Llama-3.3-70b-instruct_stage2_pred.json``) used in the paper are included in ``llamafactory_dataset.tar.gz``.

So users can directly reproduce our prediction without training/testing Stage 1 LLM first.

# Slurm Job Scirpt
Submit training jobs using ``stage1.slurm`` and ``stage2.slurm``

Key configurations: 4 A40 GPUs, 64 CPUs, 240 GB memory
- Each A40 GPU has 48 GB Memory
- Loading a 70B LLM requires ~140G CPU/GPU Memory

```bash
# sumbit job
sbatch stage1.slurm
# check whether job is runining
squeue -u $USER
# check job output
tail -f llamafactory-<JOBID>.out
```

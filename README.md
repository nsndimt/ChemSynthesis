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

2. Place all your training/evaluation and dataset description JSON files inside ``checkpoint_llamafactory`` folder; Keep the Slurm and training scripts in the repo (``LLaMA-Factory``) root.:

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
```

# Training Script

The main training pipeline is defined in ``stage1.sh`` and ``stage2.sh``. It performs:
- Model download from Hugging Face Hub
- LoRA fine-tuning with DeepSpeed
- Adapter export and LoRA merging
- Evaluation with vllm_infer.py
- Syncing checkpoints back to $HOME/checkpoint_llamafactory



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
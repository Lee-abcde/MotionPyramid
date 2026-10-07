# Python Module for MotionPyramid

This directory provides the Python training, evaluation, and inference module used in **MotionPyramid: Controllable Motion Synthesis via Stylized Phase Manifolds**.

The Python code trains the VQ phase embedding model, prepares text-to-phase metadata, trains diffusion models, evaluates checkpoints, and generates motion results for visualization in Unity.

## Python Environment Setup

The expected conda environment name is `mopyr`.

### Option A: Create from `environment.yml`

From the repository root, run:

```bash
conda env create -f Python/environment.yml
conda activate mopyr
```

If the `mopyr` environment already exists and you want to update it from the file:

```bash
conda env update --name mopyr --file Python/environment.yml --prune
conda activate mopyr
```

### Option B: Manual Setup

Use this path if `environment.yml` does not install cleanly on your machine.

```bash
conda create --name mopyr python=3.11.9 -y
conda activate mopyr
python -m pip install --upgrade pip
```

Install PyTorch. The checked-in environment uses PyTorch `2.7.1` with CUDA `11.8` wheels:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Install the project dependencies:

```bash
pip install einops==0.8.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install pandas==2.2.3
pip install blobfile==3.0.0
pip install onnxruntime-gpu==1.20.1
pip install matplotlib==3.8.4
pip install tensorboard==2.17.0
pip install omegaconf==2.3.0
pip install pytorch-lightning==2.4.0
pip install pytorch-fid==0.3.0
pip install denoising-diffusion-pytorch==2.0.17
pip install ema-pytorch==0.6.4
pip install accelerate==0.34.2
pip install safetensors==0.4.5
```

### Verify the Environment

From the repository root:

```bash
conda activate mopyr
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import clip, pandas, blobfile, einops; print('imports ok')"
```

`torch.cuda.is_available()` should print `True` for GPU training. CPU-only execution may work for lightweight preprocessing scripts, but training and generation are expected to run on CUDA.

## Data Preparation

Prepare exported motion data and text-to-phase metadata following the workflow in [`../Unity/README.md`](../Unity/README.md), especially:

- `Python/Datasets/HumanML3D`
- `Python/Datasets/HumanML3DwithRoot`
- `Python/Datasets/HumanML3DwithRoot_Text`

The text-to-phase preprocessing scripts live in:

```text
Python/utils/text2phase/HumanML3D/
```

## Stylized Manifold Training

Train the VQ phase embedding model on the HumanML3D motion dataset exported without root channels.

From the repository root:

```bash
cd Python
python train_vq.py --load=./Datasets/HumanML3D --save=./results/HumanML3D_stylized_manifold --epochs=15 --num_embed_vq=512 --kernel_size=7 --needed_channel_names "Velocities,Positions,Rotations" --fps=20 --n_latent_channel=256
```

The `--save` path controls where training logs, arguments, checkpoints, and generated phase/manifold outputs are written. You can change `./results/HumanML3D_stylized_manifold` to any experiment folder name you prefer.

If you do not train the stylized manifold model yourself, download the pretrained VQ model and per-frame phase embedding from [ETH Polybox](https://polybox.ethz.ch/index.php/s/3reE6rGw5jfpgeq). Unzip the files so the folder is available as:

```text
Python/pre-trained/HumanML3D
```

## Diffusion Model Training

The diffusion models use a pretrained VQ phase embedding model. In the commands below, `--pretrained_save=./pre-trained/HumanML3D` should point to the stylized manifold checkpoint directory, or to a copied pretrained folder with the same contents.

### Pretrained Diffusion Checkpoints

If you only want to run generation or evaluation with pretrained diffusion checkpoints, download the phase-to-motion and text-to-phase models from [Google Drive](https://drive.google.com/drive/folders/1arDtyJ3mjQpeoAO6PO-wChmPTKt-k2Op?usp=sharing). Unzip them so the checkpoint folders are available as:

```text
Python/results/HumanML3D_difftest_3
Python/results/HumanML3D_text2phase_2
```

### Phase-to-Motion Diffusion

Train the phase-to-motion diffusion model on the HumanML3D dataset exported with root channels:

```bash
cd Python
python train_mdm_2.py --unconstrained --load=./Datasets/HumanML3DwithRoot --pretrained_save=./pre-trained/HumanML3D --diff_save_dir=./results/HumanML3D_phase2motion --arch=unet --num_steps=6000_000 --save_interval=50_000 --lr=1e-4 --window=2.0 --diffusion_steps=100 --lambda_fc=0
```

### Text-to-Phase Diffusion

Train the text-to-phase diffusion model on the prepared text-to-phase dataset:

```bash
cd Python
python train_txt2phase.py --load=./Datasets/HumanML3DwithRoot_Text --dataset=text2phase --pretrained_save=./pre-trained/HumanML3D --diff_save_dir=./results/HumanML3D_text2phase --arch=unet --num_steps=6000_000 --save_interval=30_000 --lr=1e-5 --diffusion_steps=100
```

This command uses the older discrete VQ one-hot text-to-phase model: one-hot phase code, phase angle, style code, and relative root trajectory. Do not add `--use_manifold` unless you intentionally want to train the newer continuous-manifold text-to-phase variant, and keep that choice consistent during generation and evaluation.

## Inference

The following commands assume the pretrained folders from the sections above have been extracted under `Python/pre-trained/` and `Python/results/`.

### Phase-to-Motion Inference

Generate motion with the pretrained phase-to-motion diffusion checkpoint:

```bash
cd Python
python generate.py --unconstrained --load=./Datasets/HumanML3DwithRoot --pretrained_save=./pre-trained/HumanML3D --model_path=./results/HumanML3D_difftest_3/model001480000.pt --output_dir=./results/HumanML3D_difftest_3/generate/ --arch=unet --diffusion_steps=100
```

### Text-to-Phase Diffusion Inference

Generate text-conditioned phase predictions with the pretrained text-to-phase diffusion checkpoint:

```bash
cd Python
python generate_text2phase.py --load=./Datasets/HumanML3DwithRoot_Text --dataset=text2phase --pretrained_save=./pre-trained/HumanML3D --model_path=./results/HumanML3D_text2phase_2/model001680000.pt --output_dir=./results/HumanML3D_text2phase_2/generate/ --arch=unet --input_set=test --diffusion_steps=100
```

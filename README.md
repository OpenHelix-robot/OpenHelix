<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  </a>
  <a href="https://arxiv.org/abs/2505.03912">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-OpenHelix-blue">
  </a>
  <a href="https://anaconda.org/">
    <img alt="Python" src="https://img.shields.io/badge/python-3.8-blue">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/framework-PyTorch-red">
  </a>
  <a href="https://huggingface.co/OpenHelix/openhelix">
  <img alt="HF" src="https://img.shields.io/badge/HuggingFace-OpenHelix-yellow?logo=huggingface&logoColor=white">
  </a>
<!--   <a href="https://github.com/OpenHelix-robot/OpenHelix/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/OpenHelix-robot/OpenHelix?style=social">
  </a> -->
</p>

<p align="center">
  <a href="https://paperswithcode.com/sota/robot-manipulation-on-calvin?p=openhelix-a-short-survey-empirical-analysis">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openhelix-a-short-survey-empirical-analysis/robot-manipulation-on-calvin">
  </a>
</p>

# 🚀 OpenHelix: An Open-source Dual-System VLA Model for Robotic Manipulation

OpenHelix Team: 
[Can Cui*](https://cuixxx.github.io), [Pengxiang Ding*](https://dingpx.github.io), Wenxuan Song, [Shuanghao Bai](https://baishuanghao.github.io/), Xinyang Tong, Zirui Ge, Runze Suo, and others.

This is our re-implementation of [Helix](https://www.figure.ai/news/helix).

We will provide long-term maintenance for this repository.

If you have any questions, please contact us via [email](dingpx[AT]gmail.com)! 

---

# 🗞️ News
- **[2025/04]** Initial release of **OpenHelix** codebase! 🎉
- **[2025/05]** We released our paper on [arXiv](https://arxiv.org/abs/2505.03912). 📄
- **[2025/05]** We released the checkpoints of OpenHelix on Hugging Face 🤗.
- **[2025/06]** We evaluated OpenHelix on CALVIN ABC-D (EP_LEN=360), which is a mainstream setting, and found that OpenHelix achieves **SOTA** performance among dual-system VLA models. A more powerful version of OpenHelix is on the way！ — Stay tuned!
---

# 📌 TODO list
- [x] Release checkpoints for reproduction (**Scheduled Release Date: Mid-May, 2025**)
- [ ] Update the model until all effects on the robotic arm are satisfied. (**Long-term maintenance**)
- [ ] Deploying on real robots. 
- [ ] Deploying on humanoid robots.
- [ ] Realizing collaboration between humanoid robots.



---

# 🛠️ Installation

Create a conda environment with the following commands:

```bash
# Initiate conda env
conda update conda
conda create -n openhelix python=3.8 -y
conda activate openhelix

# Install CALVIN locally
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd calvin
cd calvin_env; git checkout main
cd ..
pip install setuptools==57.5.0
./install.sh; cd ..

# Clone OpenHelix repo and install
git clone git@github.com:OpenHelix-robot/OpenHelix.git
cd OpenHelix
pip install -e .

# Install diffuser
pip install diffusers["torch"]

# Install DGL (https://www.dgl.ai/pages/start.html)
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html

# Install FlashAttention (https://github.com/Dao-AILab/flash-attention#installation-and-features)
pip install packaging
pip install ninja
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

---

# 📦 Data Preparation

### Prepare data on CALVIN

* Download the play demonstrations from [Calvin](https://github.com/mees/calvin) repo.
```
> cd calvin/dataset
> sh download_data.sh ABC
```

* Package the demonstrations for training
```
> python data_preprocessing/package_calvin.py --split training
> python data_preprocessing/package_calvin.py --split validation
```

### Expected directory layout
```
./calvin/dataset/task_ABC_D
                     |------- training/
                     |------- validation/

./data/calvin/packaged_ABC_D
                     |------- training/
                     |            |------- A+0/
                     |            |          |------- ann_1.dat
                     |            |          |------- ...
                     |            |
                     |            |------- B+0/
                     |            |------- C+0/
                     |
                     |------- validation/
                                  |------- D+0/
```

---

# 🗂️ (Optional) Encode Language Instructions

We provide scripts for encoding language instructions with a CLIP Text Encoder on CALVIN.  
Alternatively, you can directly download pre-encoded instructions from [here](https://huggingface.co/katefgroup/3d_diffuser_actor/blob/main/instructions.zip).

```bash
# Encode validation instructions
python data_preprocessing/preprocess_calvin_instructions.py \
  --output instructions/calvin_task_ABC_D/validation.pkl \
  --model_max_length 16 \
  --annotation_path ./calvin/dataset/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy

# Encode training instructions
python data_preprocessing/preprocess_calvin_instructions.py \
  --output instructions/calvin_task_ABC_D/training.pkl \
  --model_max_length 16 \
  --annotation_path ./calvin/dataset/task_ABC_D/training/lang_annotations/auto_lang_ann.npy
```

---
# 📍 Checkpoints

We uploaded the model weights on Hugging Face.

| MLLM(PT) + Policy(P) | MLLM(PT) + Aux + Policy(P) |
|----------------------|-----------------------------|
| [Weights](https://huggingface.co/OpenHelix/openhelix/tree/main/prompt_tuning) | [Weights](https://huggingface.co/OpenHelix/openhelix/tree/main/prompt_tuning_aux) |

Notably, **you only need to merge the safetensors in the direcctory, e.g. "prompt_tuning_aux/llava_ckpt_safetensors", into a single pytorch_model.bin file**. Here is the code:

```
import torch
from safetensors.torch import load_file
import os

shard_folder = "/openhelix/prompt_tuning_aux/llava_ckpt_safetensors"
output_file = "/openhelix/prompt_tuning_aux/pytorch_model.bin"

shard_files = sorted([
    os.path.join(shard_folder, f)
    for f in os.listdir(shard_folder)
    if f.endswith(".safetensors")
])

merged_state_dict = {}


for shard_file in shard_files:
    shard_dict = load_file(shard_file)
    merged_state_dict.update(shard_dict)
    print(f"Loaded {shard_file} with {len(shard_dict)} tensors")


torch.save(merged_state_dict, output_file)
print(f"\nMerged model saved as: {output_file}")
```
This will generate a file named pytorch_model.bin. Copy the path of this file, along with the path to the policy.pth file in the download directory from huggingface, into the "test_trajectory_lcb_pt_act_simple_asy10.sh" script as shown below:

```
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple_asy10.py \
    --calvin_dataset_path /calvin/task_ABC_D \
    --calvin_model_path /3d_diffuser_actor/calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --calvin_gripper_loc_bounds /calvin/task_ABC_D/validation/statistics.yaml \
    --embedding_dim $C \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 0 \
    --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs_pt_1000_0324_sr1_task_latent_lcb_pt_auxin2stage_asy10/ \
    --quaternion_format $quaternion_format \
    --checkpoint /openhelix_huggingface/openhelix/prompt_tuning_aux/policy.pth \  #Here is the path of policy.pth !!!!!!!!!!!!
    --llm_ckpt /openhelix_huggingface/openhelix/prompt_tuning_aux  #Here is the path of pytorch_model.bin !!!!!!!!!!!!!!!!!!!!
```
**The --checkpoint argument should be set to the path of policy.pth, and the --llm_ckpt argument should be set to the path of pytorch_model.bin.**

The results on CALVIN ABC-D. MLLM (PT) denotes our proposed prompt tuning method for MLLM training. Policy(P) indicates loading from a pretrained policy model. Asy(10) represents inference with a 10-step time delay. AUX denotes the additionally introduced auxiliary tasks.

| Method                                                   |   1   |   2   |   3   |   4   |   5   | Avg. Len. ↑ |
|----------------------------------------------------------|-------|-------|-------|-------|-------|--------------|
| Only Policy                                              | 92.2  | 78.7  | 63.9  | 51.2  | 41.2  | 3.27         |
| MLLM (PT) + Policy(P) (EP_LEN=60)                        | 92.2  | 79.2  | 65.0  | 52.9  | 40.9  | 3.30         |
| MLLM (PT) + AUX + Policy(P) + Asy(10) (EP_LEN=60)        | 93.3  | 81.8  | 67.9  | 56.6  | 46.0  | 3.45         |
| MLLM (PT) + Policy(P) (EP_LEN=360)                       | 96.3  | 87.3  | 77.5  | 66.5  | 55.5  | 3.83         |
| MLLM (PT) + AUX + Policy(P) + Asy(10) (EP_LEN=360)       | 97.1  | 91.4  | 82.8  | 72.6  | 64.1  | **4.08**     |
| Robodual                                                 | 94.4  | 82.7  | 72.1  | 62.4  | 54.4  | 3.66         |
| UniVLA                                                   | 95.5  | 85.8  | 75.4  | 66.9  | 56.5  | 3.80         |
| Seer                                                     | 94.4  | 87.2  | 79.9  | 72.2  | 64.3  | 3.98         |
| GR-MG                                                    | 96.8  | 89.3  | 81.5  | 72.7  | 64.4  | 4.04         |

# 🎮 Getting Started

### Train Openhelix on CALVIN:
```
> bash scripts/train_trajectory_lcb_pt_act_simple.sh
```
### To evaluate pre-trained weights:
  - First, download the weights and place them under `train_logs/`.
  - Next, you can run the provided evaluation script.
```
> bash scripts/test_trajectory_lcb_pt_act_simple_asy10.sh
```
---

# 📚 Citation

If you find this code useful for your research, please consider citing our paper.

```bibtex
@article{cui2025openhelix,
  title={OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation},
  author={Cui, Can and Ding, Pengxiang and Song, Wenxuan and Bai, Shuanghao and Tong, Xinyang and Ge, Zirui and Suo, Runze and Zhou, Wanqi and Liu, Yang and Jia, Bofang and others},
  journal={arXiv preprint arXiv:2505.03912},
  year={2025}
}
```

---

# 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

# 🙏 Acknowledgement

Parts of this codebase are adapted from:
- [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)
- [CALVIN](https://github.com/mees/calvin)

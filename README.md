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
  <a href="https://github.com/OpenHelix-robot/OpenHelix/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/OpenHelix-robot/OpenHelix?style=social">
  </a>
</p>

# 🚀 OpenHelix: An Open-source Dual-System VLA Model for Robotic Manipulation

OpenHelix Team: 
[Can Cui*](https://cuixxx.github.io), [Pengxiang Ding*](https://dingpx.github.io), Wenxuan Song, [Shuanghao Bai](https://github.com/BaiShuanghao), Xinyang Tong, Zirui Ge, Runze Suo and others.

This is our re-implementation of [Helix](https://www.figure.ai/news/helix).

We will provide long-term maintenance for this repository.

If you have any questions, please contact us via [email](dingpx[AT]gmail.com)! 

---

# 🗞️ News
- **[2025/04]** Initial release of **OpenHelix** codebase! 🎉
- **[2025/05]** We released our paper on [arXiv](https://arxiv.org/abs/2505.03912). 📄
- **[2025/05]** We released the checkpoints of OpenHelix on Hugging Face 🤗.
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

The results on CALVIN ABC-D. MLLM (PT) denotes our proposed prompt tuning method for MLLM training. Policy(P) indicates loading from a pretrained policy model. Asy(10) represents inference with a 10-step time delay. AUX denotes the additionally introduced auxiliary tasks.

| Method                                                   |   1   |   2   |   3   |   4   |   5   | Avg. Len. ↑ |
|----------------------------------------------------------|-------|-------|-------|-------|-------|--------------|
| Only Policy                                              | 92.2  | 78.7  | 63.9  | 51.2  | 41.2  | 3.27         |
| MLLM (PT) + Policy(P) (EP_LEN=60)                        | 92.2  | 79.2  | 65.0  | 52.9  | 40.9  | 3.30         |
| MLLM (PT) + AUX + Policy(P) + Asy(10) (EP_LEN=60)        | 93.3  | 81.8  | 67.9  | 56.6  | 46.0  | 3.45         |
| MLLM (PT) + Policy(P) (EP_LEN=360)                       | 96.3  | 87.3  | 77.5  | 66.5  | 55.5  | 3.83         |
| MLLM (PT) + AUX + Policy(P) + Asy(10) (EP_LEN=360)       | 97.1  | 91.4  | 82.8  | 72.6  | 64.1  | **4.08**     |

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

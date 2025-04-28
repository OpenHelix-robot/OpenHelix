<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  </a>
  <a href="https://arxiv.org">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-OpenHelix-blue">
  </a>
  <a href="https://anaconda.org/">
    <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/framework-PyTorch-red">
  </a>
  <a href="https://github.com/Cuixxx/OpenHelix/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/Cuixxx/OpenHelix?style=social">
  </a>
</p>

# 🚀 OpenHelix: An Open-source Dual-System VLA Model for Robotics Manipulation
By [Can Cui*](https://cuixxx.github.io), [Pengxiang Ding*](https://dingpx.github.io), and Wenxuan Song.  

This is our re-implementation of [Helix](https://www.figure.ai/news/helix).

---

# 🗞️ News
- **[2025/04]** Initial release of **OpenHelix** codebase! 🎉
- **[2025/04]** We released our paper on [arXiv](https://arxiv.org). 📄

---

# 📌 TODO list
- [ ] Release checkpoints for reproduction (**Scheduled Release Date: Mid-April, 2025**)

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
./install.sh; cd ..

# Clone OpenHelix repo and install
git clone https://github.com/Cuixxx/OpenHelix.git
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

# 🎮 Getting Started

- See [Getting Started with CALVIN](./docs/GETTING_STARTED_CALVIN.md) for a full tutorial!
- To evaluate pre-trained weights:
  - First, download the weights and place them under `train_logs/`.
  - For CALVIN experiments, you can run the provided [test_trajectory_calvin.sh](./scripts/test_trajectory_calvin.sh) script.

---

# 📚 Citation

If you find this code useful for your research, please consider citing our paper.

---

# 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

# 🙏 Acknowledgement

Parts of this codebase are adapted from:
- [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)
- [CALVIN](https://github.com/mees/calvin)

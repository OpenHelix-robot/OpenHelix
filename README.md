# 🚀 OpenHelix: An Open-source Dual-System VLA Model for Robotics Manipulation
By [Can Cui*](https://cuixxx.github.io), [Pengxiang Ding*](https://dingpx.github.io), and Wenxuan Song.  

This is our re-implementation of [Helix](https://www.figure.ai/news/helix).

---

# 🗞️ News
- **[2025/04]** Initial release of **OpenHelix** codebase! 🎉
- **[2024/10]** We released our paper on [arXiv](https://arxiv.org). 📄

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

# Clone OpenHelix repo and install
git clone https://github.com/Cuixxx/OpenHelix.git
cd OpenHelix
pip install -e .

# Install diffuser
pip install diffusers["torch"]

# Install DGL (https://www.dgl.ai/pages/start.html)
pip install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl

# Install FlashAttention (https://github.com/Dao-AILab/flash-attention#installation-and-features)
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

---

# 📦 Data Preparation

See [Preparing CALVIN dataset](./docs/DATA_PREPARATION_CALVIN.md).

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
- [Act3D](https://github.com/zhouxian/act3d-chained-diffuser)
- [CALVIN](https://github.com/mees/calvin)

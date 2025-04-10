# OpenHelix: An Open-source Dual-System VLA Model for Robotics Manipulation
By [Can Cui*](https://cuixxx.github.io), [Pengxiang Ding*](https://dingpx.github.io), and Wenxuan Song.

This is our re-implementation of the [Helix](https://www.figure.ai/news/helix).

<!-- ![teaser](https://3d-diffuser-actor.github.io/static/videos/3d_scene.mp4) -->
<!-- ![teaser](figure5_3.pdf) -->


# Installation
Create a conda environment with the following command:

```
# initiate conda env
> conda update conda
> conda create -n openhelix python=3.8 -y
> conda activate openhelix

# Clone openhelix repo and pip install to download dependencies
> git clone https://github.com/Cuixxx/OpenHelix.git
> cd OpenHelix
> pip install -e .

# install diffuser
> pip install diffusers["torch"]

# install dgl (https://www.dgl.ai/pages/start.html)
>  pip install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl

# install flash attention (https://github.com/Dao-AILab/flash-attention#installation-and-features)
> pip install packaging
> pip install ninja
> pip install flash-attn --no-build-isolation
```

### Install CALVIN locally

Remember to use the latest `calvin_env` module, which fixes bugs of `turn_off_led`.  See this [post](https://github.com/mees/calvin/issues/32#issuecomment-1363352121) for detail.
```
> git clone --recurse-submodules https://github.com/mees/calvin.git
> export CALVIN_ROOT=$(pwd)/calvin
> cd calvin
> cd calvin_env; git checkout main
> cd ..
> ./install.sh; cd ..
```

# Data Preparation

See [Preparing CALVIN dataset](./docs/DATA_PREPARATION_CALVIN.md).


### (Optional) Encode language instructions

3dda has provided thier scripts for encoding language instructions with CLIP Text Encoder on CALVIN.  Otherwise, you can find the encoded instructions on CALVIN and RLBench ([Link](https://huggingface.co/katefgroup/3d_diffuser_actor/blob/main/instructions.zip)).
```
> python data_preprocessing/preprocess_calvin_instructions.py --output instructions/calvin_task_ABC_D/validation.pkl --model_max_length 16 --annotation_path ./calvin/dataset/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy

> python data_preprocessing/preprocess_calvin_instructions.py --output instructions/calvin_task_ABC_D/training.pkl --model_max_length 16 --annotation_path ./calvin/dataset/task_ABC_D/training/lang_annotations/auto_lang_ann.npy
```

### Evaluate the pre-trained weights
First, donwload the weights and put under `train_logs/`
* For CALVIN, you can run [this bashcript](./scripts/test_trajectory_calvin.sh).

# Getting started

See [Getting started with CALVIN](./docs/GETTING_STARTED_CALVIN.md).


# Citation
If you find this code useful for your research, please consider citing our paper.

# License
This code base is released under the MIT License (refer to the LICENSE file for details).

# Acknowledgement
Parts of this codebase have been adapted from [Act3D](https://github.com/zhouxian/act3d-chained-diffuser) and [CALVIN](https://github.com/mees/calvin).

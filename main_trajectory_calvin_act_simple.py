"""Main script for trajectory optimization."""

import os
import random
import pickle

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np

from datasets.dataset_calvin import CalvinDataset
from main_trajectory_act_simple import TrainTester as BaseTrainTester
from main_trajectory_act_simple import traj_collate_fn, fig_to_numpy, Arguments
from utils.common_utils import (
    load_instructions, get_gripper_loc_bounds
)


def load_instructions(instruction, split):
    instructions = pickle.load(
        open(f"{instruction}/{split}.pkl", "rb")
    )['embeddings']
    instructions_text = pickle.load(
        open(f"{instruction}/{split}.pkl", "rb")
    )['text']
    return instructions, instructions_text


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        train_instruction, train_text = load_instructions(
            self.args.instructions, 'training'
        )
        test_instruction, val_text = load_instructions(
            self.args.instructions, 'validation'
        )
        taskvar = [
            ("A", 0), ("B", 0), ("C", 0), ("D", 0),
        ]

        # Initialize datasets with arguments
        train_dataset = CalvinDataset(
            root=self.args.dataset,
            instructions=train_instruction,
            instructions_text=train_text,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            relative_action=bool(self.args.relative_action)
        )
        test_dataset = CalvinDataset(
            root=self.args.valset,
            instructions=test_instruction,
            instructions_text=val_text,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size_val,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            relative_action=bool(self.args.relative_action)
        )
        return train_dataset, test_dataset

    def save_checkpoint(self, model, optimizer, step_id, best_loss):
        """Save checkpoint if requested."""
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / '{:07d}.pth'.format(step_id))
        return best_loss

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer


def generate_visualizations(pred, gt, mask, box_size=0.05):
    batch_idx = 0
    images = []
    for batch_idx in range(min(pred.shape[0], 5)):
        cur_pred = pred[batch_idx].detach().cpu().numpy()
        cur_gt = gt[batch_idx].detach().cpu().numpy()
        cur_mask = mask[batch_idx].detach().cpu().numpy()

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            cur_pred[~cur_mask][:, 0],
            cur_pred[~cur_mask][:, 1],
            cur_pred[~cur_mask][:, 2],
            color='red', label='pred'
        )
        ax.scatter3D(
            cur_gt[~cur_mask][:, 0],
            cur_gt[~cur_mask][:, 1],
            cur_gt[~cur_mask][:, 2],
            color='blue', label='gt'
        )

        center = cur_gt[~cur_mask].mean(0)
        ax.set_xlim(center[0] - box_size, center[0] + box_size)
        ax.set_ylim(center[1] - box_size, center[1] + box_size)
        ax.set_zlim(center[2] - box_size, center[2] + box_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.legend()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        img = fig_to_numpy(fig, dpi=120)
        plt.close()
        images.append(img)
    images = np.concatenate(images, axis=1)
    return images.transpose(2, 0, 1)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)

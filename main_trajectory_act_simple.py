"""Main script for trajectory optimization."""

import io
import os
from pathlib import Path
import random
from typing import Tuple, Optional

import cv2
from matplotlib import pyplot as plt
import numpy as np
import tap
import torch
import torch.distributed as dist
from torch.nn import functional as F

from datasets.dataset_engine import RLBenchDataset
from engine_act_simple import BaseTrainTester
from diffuser_actor import DiffuserActorACTS

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)

from datasets.calvin_dataset import transfer
from planer_utils import input_processing_real_batch
import time


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0
    training_checkpoint: Optional[Path] = None
    
    # LISA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"
    llava_dir: str = "/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1"
    vision_tower: str = "/clip-vit-large-patch14"
    sample_rate: int=1

    # Training and validation datasets
    dataset: Path
    valset: Path
    dense_interpolation: int = 0
    interpolation_length: int = 100

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    wd: float = 5e-3  # used only for CALVIN
    train_iters: int = 200_000
    stage2_train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'wxyz'
    diffusion_timesteps: int = 100
    keypose_only: int = 0
    num_history: int = 0
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = load_instructions(
            self.args.instructions,
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]

        # Initialize datasets with arguments
        train_dataset = RLBenchDataset(
            root=self.args.dataset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.stage2_train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length
        )
        test_dataset = RLBenchDataset(
            root=self.args.valset,
            instructions=instruction,
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
            interpolation_length=self.args.interpolation_length
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffuserActorACTS(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced)
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()
    
    def train_one_step(self, model, criterion, optimizer, step_id, sample, embedding, lmcliploss=None, act_pred=None):
        """Run a single training step."""
        if self.args.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        else:
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

        # Forward pass
        curr_gripper = (
            sample["curr_gripper"] if self.args.num_history < 1
            else sample["curr_gripper_history"][:, -self.args.num_history:]
        )
        aux_loss = 0
        if (step_id + 1) < self.args.train_iters:
            out = model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                embedding,
                curr_gripper
            )
        else:
            out, aux_loss = model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                embedding,
                curr_gripper,
                act_pred
            )
        if (step_id + 1) < self.args.train_iters:
            model.requires_grad_(False)
        else:
            model.requires_grad_(True)

        action_loss = criterion.compute_loss(out)
        if lmcliploss is not None:
            loss = action_loss + lmcliploss
        else:
            loss = action_loss

        loss.backward()

        # Update
        if ((step_id + 1) > self.args.train_iters) and ((step_id + 1) % self.args.accumulate_grad_batches == 0):
            optimizer.step()
            optimizer.zero_grad()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % (0.01 * self.args.val_freq) == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("action_loss", aux_loss, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)
            

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters, total_action_embedding,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                total_action_embedding.to(device),
                # sample["instr"].to(device),
                curr_gripper.to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)
    
    @torch.no_grad()
    def evaluate_nsteps_lcb(self, model, criterion, loader, step_id, LCB_model, clip_image_processor, tokenizer, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break
            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
            conversations, questions = transfer(sample['instr_text'])
            start_idx = []
            initial_id=0
            for x in range(len(conversations)):
                if sample['curr_gripper_history'][x][0][0]==sample['curr_gripper_history'][x][2][0]:
                    initial_id = x
                if (x-initial_id) % self.args.sample_rate == 0:
                    start_idx.append(x)
            image_rgb = sample['rgbs'][:, 0]
            image_select_rows = image_rgb[start_idx]
            conv_select = [questions[y] for y in start_idx]
            print("conv_select", conv_select)
            image_clip, image, input_ids, attention_masks, targets = input_processing_real_batch(image_tensor=image_select_rows, conv_list=conv_select, clip_image_processor=clip_image_processor, tokenizer=tokenizer)
            LCB_model.eval()
        
            output_ids, pred_actions_embedding = LCB_model.module.evaluate(image_clip, input_ids, max_new_tokens=512, tokenizer=tokenizer)
            print(pred_embeddings.shape)

            for z in range(len(conversations)):
                if z == 0:
                    total_action_embedding[z] = pred_actions_embedding[start_idx.index(0)]
                else:
                    previous_idx = 0
                    total_action_embedding[z] = total_action_embedding[previous_idx]
            total_action_embedding = total_action_embedding.unsqueeze(1)

############second stage
            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                total_action_embedding.to(device),
                # sample["instr"].to(device),
                curr_gripper.to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)

    

def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr"
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }
    
    ret_dict["instr_text"] = []
    ret_dict["task"] = []
    for item in batch:
        ret_dict["instr_text"] += item['instr_text']
        ret_dict["task"] += item['task']
    return ret_dict


class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, mask=None, is_loss=True):
        if not is_loss:
            assert gt is not None and mask is not None
            return self.compute_metrics(pred, gt, mask)[0]['action_mse']
        return pred

    @staticmethod
    def compute_metrics(pred, gt, mask):
        # pred/gt are (B, L, 7), mask (B, L)
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        # symmetric quaternion eval
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        # gripper openess
        openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        tr = 'traj_'

        # Trajectory metrics
        ret_1, ret_2 = {
            tr + 'action_mse': F.mse_loss(pred, gt),
            tr + 'pos_l2': pos_l2.mean(),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
            tr + 'rot_l1': quat_l1.mean(),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
            tr + 'gripper': openess.flatten().float().mean()
        }, {
            tr + 'pos_l2': pos_l2.mean(-1),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
            tr + 'rot_l1': quat_l1.mean(-1),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
        }

        # Keypose metrics
        pos_l2 = ((pred[:, -1, :3] - gt[:, -1, :3]) ** 2).sum(-1).sqrt()
        quat_l1 = (pred[:, -1, 3:7] - gt[:, -1, 3:7]).abs().sum(-1)
        quat_l1_ = (pred[:, -1, 3:7] + gt[:, -1, 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        ret_1.update({
            'pos_l2_final': pos_l2.mean(),
            'pos_l2_final<0.01': (pos_l2 < 0.01).float().mean(),
            'rot_l1': quat_l1.mean(),
            'rot_l1<0025': (quat_l1 < 0.025).float().mean()
        })
        ret_2.update({
            'pos_l2_final': pos_l2,
            'pos_l2_final<0.01': (pos_l2 < 0.01).float(),
            'rot_l1': quat_l1,
            'rot_l1<0.025': (quat_l1 < 0.025).float(),
        })

        return ret_1, ret_2


def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def generate_visualizations(pred, gt, mask, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()
    mask = mask[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
        color='red', label='pred'
    )
    ax.scatter3D(
        gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
        color='blue', label='gt'
    )

    center = gt[~mask].mean(0)
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
    return img.transpose(2, 0, 1)


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

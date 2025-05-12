"""Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""
import os
import gc
from typing import Tuple, Optional, List
import random
import logging
from pathlib import Path

import tap
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import yaml
from tqdm import tqdm

from utils.common_utils import get_gripper_loc_bounds
from online_evaluation_calvin.evaluate_model import create_model
from online_evaluation_calvin.evaluate_utils import (
    prepare_visual_states,
    prepare_proprio_states,
    count_success,
    get_env_state_for_initial_condition,
    collect_results,
    write_results,
    get_log_dir
)
from online_evaluation_calvin.multistep_sequences import get_sequences
from online_evaluation_calvin.evaluate_utils import get_env

logger = logging.getLogger(__name__)

EP_LEN = 60
NUM_SEQUENCES = 100
EXECUTE_LEN = 20

import ssl
ssl._create_default_https_context=ssl._create_unverified_context


class Arguments(tap.Tap):
    # Online enviornment
    calvin_dataset_path: Path = "/home/tsungwek/repos/calvin/dataset/task_ABC_D"
    calvin_model_path: Path = "/home/tsungwek/repos/calvin/calvin_models"
    calvin_demo_tasks: Optional[List[str]] = None
    device: str = "cuda"
    text_encoder: str = "clip"
    text_max_length: int = 16
    save_video: int = 0
    task_name: str = 'rotate_blue_block_right'
    videos_dynamics_path: str = '/'

    # Offline data loader
    seed: int = 0
    tasks: Tuple[str, ...] # indicates the environment
    checkpoint: Path
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    calvin_gripper_loc_bounds: Optional[str] = None
    relative_action: int = 0

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs" / "calvin"

    # Model
    action_dim: int = 7 # dummy, as DiffuserActor assumes action_dim is 7
    image_size: str = "256,256" # decides the FPN architecture
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'wxyz'
    diffusion_timesteps: int = 100
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 3
    num_history: int = 0
    interpolation_length: int = 2 # the number of steps to reach keypose


def make_env(dataset_path, show_gui=True, split="validation", scene=None):
    val_folder = Path(dataset_path) / f"{split}"
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env

def write_results_updadte(log_dir, seq_ind, success_aggregators):
    """Write the number of completed tasks of each instruction chain to a file.
    """
    with open(log_dir / f"result.txt", "a") as write_file:
        write_file.write(f"{seq_ind} {success_aggregators[0]} {success_aggregators[1]} {success_aggregators[2]} {success_aggregators[3]} {success_aggregators[4]}\n")


def collect_results_update(log_dir):
    """Load the number of completed tasks of each instruction chain from a file.
    """
    if os.path.isfile(str(Path(log_dir) / "result.txt")):
        with open(str(Path(log_dir) / "result.txt")) as f:
            lines = f.read().split("\n")[:-1]
    else:
        lines = []

    results, seq_inds = [], []
    for line in lines:
        seq, res0, res1, res2, res3, res4= line.split(" ")
        results.append([int(res0),int(res1),int(res2),int(res3),int(res4)])
        seq_inds.append(int(seq))

    return results, seq_inds

def evaluate_policy(model, env, conf_dir, eval_log_dir=None, save_video=False,
                    sequence_indices=[],task_name=None,videos_dynamics_path=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: an instance of CalvinBaseModel
        env: an instance of CALVIN_ENV
        conf_dir: Path to the directory containing the config files of CALVIN
        eval_log_dir: Path where to log evaluation results
        save_video: a boolean indicates whether to save the video
        sequence_indices: a list of integers indicates the indices of the
            instruction chains to evaluate

    Returns:
        results: a list of integers indicates the number of tasks completed
    """
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results, tested_sequence_indices = collect_results_update(eval_log_dir)
    
    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        
        success_aggregators = [0,0,0,0,0]
        result, success_aggregators = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, val_annotations, save_video, seq_ind, success_aggregators,task_name,videos_dynamics_path
        )
        results.append(success_aggregators)
        write_results_updadte(eval_log_dir, seq_ind, success_aggregators)


    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      val_annotations, save_video, seq_ind, success_aggregators,task_name,videos_dynamics_path):
    """
    Evaluates a sequence of language instructions.

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_checker: an indicator of whether the current task is completed
        initial_state: a tuple of `robot_obs` and `scene_obs`
            see: https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation
        eval_sequence: a list indicates the instruction chain
        val_annotations: a dictionary of task instructions
        save_video: a boolean indicates whether to save the video

    Returns:
        success_counter: an integer indicates the number of tasks completed
        video_aggregator: a list of lists of images that shows the trajectory
            of the robot

    """
    if 'blue_block' in task_name:
        initial_state = {'led': 1, 'lightbulb': 1, 'slider': 'left', 'drawer': 'open', 'red_block': 'slider_left', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
        eval_sequence = [task_name,task_name,task_name,task_name,task_name]
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

    init_blueblock_pose = scene_obs[12:18]
     
    
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    aug_method = 0
    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        print("aug_method before:",aug_method)
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        success, video = rollout(env, model, task_checker,
                                 subtask, lang_annotation, init_blueblock_pose,seq_ind,aug_method,robot_obs,scene_obs)
        aug_method +=1
        print("aug_method after:",aug_method)
        import moviepy.video.io.ImageSequenceClip
        from moviepy.editor import vfx
        clip = []
        import cv2
        for img_ind, img in enumerate(video):
            cv2.putText(img.astype(np.uint8),
                        f'{seq_ind}: {subtask}',
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        (0, 0, 0),
                        1,
                        2)
            video[img_ind] = img
        clip.extend(video)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
        if success:
            flag = 'Succ'
        else:
            flag = 'Fail'
        clip.write_videofile(f"{videos_dynamics_path}/{task_name}/{task_name}_seq{seq_ind}_aug{aug_method-1}_{flag}.mp4")
        video_aggregators.append(video)

        if success:
            success_counter += 1
            success_aggregators[aug_method-1] += 1
        # print('success: ', success_aggregators)

    return success_counter, success_aggregators


def rollout(env, model, task_oracle, subtask, lang_annotation,init_blueblock_pose,seq_ind,aug_method,robot_obs,scene_obs):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_oracle: an indicator of whether the current task is completed
        subtask: a string indicates the task name
        lang_annotation: a string indicates the instruction of the task

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    video = [] # show video for debugging
    obs = env.get_obs()
    
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)


    model.reset()
    start_info = env.get_info()

    print('------------------------------')
    print(f'task: {lang_annotation}')
    video.append(obs["rgb_obs"]["rgb_static"])
    contact_flag = False
    pbar = tqdm(range(EP_LEN))
    for step in pbar:
        obs = prepare_visual_states(obs, env)
        obs = prepare_proprio_states(obs, env)
        lang_embeddings = model.encode_instruction(lang_annotation, model.args.device)
        with torch.cuda.amp.autocast():
            trajectory = model.step(obs, lang_embeddings)
        for act_ind in range(min(trajectory.shape[1], EXECUTE_LEN)):
            # calvin_env executes absolute action in the format of:
            # [[x, y, z], [euler_x, euler_y, euler_z], [open]]
            curr_action = [
                trajectory[0, act_ind, :3],
                trajectory[0, act_ind, 3:6],
                trajectory[0, act_ind, [6]]
            ]
            pbar.set_description(f"step: {step}")
            curr_proprio = obs['proprio']
            obs, _, _, current_info = env.step(curr_action)
            obs['proprio'] = curr_proprio
            
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )
            
            contact_instruct = 'contact_blue_block'
            current_contact_info = task_oracle.get_task_info_for_set(start_info, current_info, {contact_instruct})
            if len(current_contact_info) > 0:
                print('contacted:',current_contact_info)
                contact_flag = True
            if not contact_flag:
                h = init_blueblock_pose[0]
                k = init_blueblock_pose[1]
                h_r = 0.00
                k_r = 0.09
                r = 0.03
                if aug_method == 1:
                    update_pose_x = h - (h - h_r) / EP_LEN * (step+1)
                    update_pose_y = k
                elif aug_method == 2:
                    update_pose_x = h
                    update_pose_y = k - (k - k_r) / EP_LEN * (step+1)
                elif aug_method == 3:
                    update_pose_x = h - (h - h_r) / EP_LEN * (step+1)
                    update_pose_y = k - (k - k_r) / EP_LEN * (step+1)
                elif aug_method == 4:
                    # 生成圆上的点
                    theta = np.linspace(0, 2 * np.pi, EP_LEN)
                    x = h + r * np.cos(theta)
                    y = k + r * np.sin(theta)
                    update_pose_x = x[step]
                    update_pose_y = y[step]
                else:
                    update_pose_x = h
                    update_pose_y = k
                    
                scene_obs = obs['scene_obs']
                scene_obs[12] = update_pose_x
                scene_obs[13] = update_pose_y
                env.reset(robot_obs=obs['robot_obs'], scene_obs=scene_obs)

            video.append(obs["rgb_obs"]["rgb_static"])
            

            if len(current_task_info) > 0:
                return True, video
    
    return False, video


def get_calvin_gripper_loc_bounds(args):
    with open(args.calvin_gripper_loc_bounds, "r") as stream:
       bounds = yaml.safe_load(stream)
       min_bound = bounds['act_min_bound'][:3]
       max_bound = bounds['act_max_bound'][:3]
       gripper_loc_bounds = np.stack([min_bound, max_bound])

    return gripper_loc_bounds


def main(args):

    # These location bounds are extracted from language-annotated episodes
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )

    # These location bounds are extracted from every episode in play trajectory
    if args.calvin_gripper_loc_bounds is not None:
        args.calvin_gripper_loc_bounds = get_calvin_gripper_loc_bounds(args)

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # evaluate a custom model
    model = create_model(args)

    sequence_indices = [
        i for i in range(args.local_rank, NUM_SEQUENCES, int(os.environ["WORLD_SIZE"]))
    ]

    env = make_env(args.calvin_dataset_path, show_gui=False)
    evaluate_policy(model, env,
                    conf_dir=Path(args.calvin_model_path) / "conf",
                    eval_log_dir=args.base_log_dir,
                    sequence_indices=sequence_indices,
                    save_video=args.save_video,
                    task_name=args.task_name,
                    videos_dynamics_path=args.videos_dynamics_path)

    results, sequence_inds = collect_results_update(args.base_log_dir)
    
    def count_success_update(results):
        sum_results = [sum(sublist[i] for sublist in results) for i in range(len(results[0]))]
        step_success = []
        for i in range(1, 6):
            n_success = sum_results[i-1]
            sr = n_success / len(results)
            step_success.append(sr)
        return step_success
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
        for i, v in enumerate(count_success_update(results))]) + "|"
    )
    print(f'Load {len(results)}/{NUM_SEQUENCES} episodes...')
    print(str_results + "\n")

    del env
    gc.collect()
    
def Model_init(args):
    
    torch_dtype = torch.bfloat16
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.llava_dir, cache_dir=None, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("<ACT>")
    seg_token_idx = tokenizer("<ACT>", add_special_tokens=False).input_ids[0]

    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    model_args = {
        "out_dim": 512,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": True,
    }
    
    model = LISAForCausalLM.from_pretrained(args.llava_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args).cuda()
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=0)
    
    return clip_image_processor, tokenizer, model
    
def input_processing_real_batch(image_tensor, conv_list, clip_image_processor, tokenizer):
    '''
    preprocess input (image/text)
    '''
    images = np.expand_dims(image_tensor, axis=0)
    images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    image_clip_batch = clip_image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"]
    image_clip_batch = image_clip_batch.to(torch.bfloat16).cuda()

    from model.llava import conversation as conversation_lib
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v1']
    conv = conversation_lib.default_conversation.copy()
    sep = conv.sep + conv.roles[1] + ":" 
    short_input_ids = []
    for conversation in conv_list:
        rounds = conversation.split(conv.sep2)
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            parts[0] += (sep+" "+"<ACT>")
            if DEFAULT_IMAGE_TOKEN in conversation:
                short_input_ids.append(tokenizer_image_token(parts[0], tokenizer, return_tensors="pt"))
            else:
                short_input_ids.append(tokenizer(parts[0]).input_ids)
    input_ids = torch.nn.utils.rnn.pad_sequence(short_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()  # 相同 batch 中进行补齐
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()
    targets[:,:] = IGNORE_INDEX

    truncate_len = tokenizer.model_max_length - 255
    
    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]

    return image_clip_batch, input_ids, attention_masks, targets

if __name__ == "__main__":
    args = Arguments().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)

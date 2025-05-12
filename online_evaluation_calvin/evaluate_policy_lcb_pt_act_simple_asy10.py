"""Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""
import os
import gc
from typing import Tuple, Optional, List
import random
import logging
from pathlib import Path
import time
import tap
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import yaml
from tqdm import tqdm

from utils.common_utils import get_gripper_loc_bounds
from online_evaluation_calvin.evaluate_model_act import create_model
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


import os
import argparse
import transformers
from transformers import CLIPImageProcessor
import cv2
import torch
import torch.nn.functional as F
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)
from peft import PeftModel
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX)

from peft import LoraConfig, get_peft_model
from model.llava import conversation as conversation_lib
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from planer import LISAForCausalLM
from datasets.calvin_dataset import transfer
from torchvision import transforms
from PIL import Image
import json
logger = logging.getLogger(__name__)

EP_LEN = 60
NUM_SEQUENCES = 1000
EXECUTE_LEN = 20


class Arguments(tap.Tap):
    # Online enviornment
    calvin_dataset_path: Path = "/home/tsungwek/repos/calvin/dataset/task_ABC_D"
    calvin_model_path: Path = "/home/tsungwek/repos/calvin/calvin_models"
    calvin_demo_tasks: Optional[List[str]] = None
    device: str = "cuda"
    text_encoder: str = "clip"
    text_max_length: int = 16
    save_video: int = 0

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
    
    #LLM
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"
    llava_dir: str = "/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1"
    vision_tower: str = "/clip-vit-large-patch14"
    llm_ckpt: str = ''
    

def make_env(dataset_path, show_gui=True, split="validation", scene=None):
    val_folder = Path(dataset_path) / f"{split}"
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env


def evaluate_policy(model, LLM_model, clip_image_processor, tokenizer, env, conf_dir, eval_log_dir=None, save_video=False,
                    sequence_indices=[]):
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

    results, tested_sequence_indices = collect_results(eval_log_dir)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        result, videos = evaluate_sequence(
            env, model, LLM_model, clip_image_processor, tokenizer, task_oracle, initial_state,
            eval_sequence, val_annotations, save_video
        )
        write_results(eval_log_dir, seq_ind, result)
        results.append(result)
        str_results = (
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
            for i, v in enumerate(count_success(results))]) + "|"
        )
        print(str_results + "\n")

        if save_video:
            import moviepy.video.io.ImageSequenceClip
            from moviepy.editor import vfx
            clip = []
            import cv2
            for task_ind, (subtask, video) in enumerate(zip(eval_sequence, videos)):
                for img_ind, img in enumerate(video):
                    img = np.ascontiguousarray(img)
                    cv2.putText(img,
                                f'{task_ind}: {subtask}',
                                (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                (0, 0, 0),
                                1,
                                2)
                    video[img_ind] = img
                clip.extend(video)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
            clip.write_videofile(f"calvin_seq{seq_ind}.mp4")


    return results


def evaluate_sequence(env, model, LLM_model, clip_image_processor, tokenizer, task_checker, initial_state, eval_sequence,
                      val_annotations, save_video):
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
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        # lang_annotation = random.choice(val_annotations[subtask])
        success, video = rollout(env, model, LLM_model, clip_image_processor, tokenizer, task_checker,
                                 subtask, lang_annotation)
        video_aggregators.append(video)

        if success:
            success_counter += 1
        else:
            return success_counter, video_aggregators
    return success_counter, video_aggregators


def rollout(env, model, LLM_model, clip_image_processor, tokenizer, task_oracle, subtask, lang_annotation):#这是运行试验的主要代码
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

    model.reset()
    start_info = env.get_info()

    print('------------------------------')
    print(f'task: {lang_annotation}')
    video.append(obs["rgb_obs"]["rgb_static"])

    pbar = tqdm(range(EP_LEN))
    LLM_model.eval()

    for step in pbar:
        obs = prepare_visual_states(obs, env)
        obs = prepare_proprio_states(obs, env)
        
        # import pdb; pdb.set_trace()
        #lang_annotation='push the sliding door to the right side'
        text_list = [lang_annotation]
        # import pdb; pdb.set_trace()
        conversations, questions = transfer(text_list)
        # convs_select = conversations[0]
        #img:array(200, 200, 3)这个后面扩充了第0维度 conv:list，长度与img的batch一致
        #"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nCan you control the robot to push the sliding door to the right side? ASSISTANT: Sure, I will push the sliding door to the right side <ACT>.</s>"
        image_clip, input_ids, attention_masks, targets = input_processing_real_batch(image_tensor=obs["rgb_obs"]["rgb_static"], conv_list=conversations, clip_image_processor=clip_image_processor, tokenizer=tokenizer)
        # import pdb; pdb.set_trace()
        if step%1 == 0:
            output_ids, pred_embeddings = LLM_model.evaluate(image_clip, input_ids, attention_masks)#input_ids.size()=torch.Size([333, 3])
            # latent_embs.append(pred_embeddings)
            # lang_embeddings = pred_embeddings.unsqueeze(0)#[1, 512]-->[1, 1, 512]
            lang_embeddings = pred_embeddings.unsqueeze(0)
        # output_ids, pred_embeddings = LLM_model.evaluate(image_clip, input_ids, attention_masks)#input_ids.size()=torch.Size([333, 3])
        # print(pred_embeddings.shape)
        # lang_embeddings = pred_embeddings.unsqueeze(0)#[1, 512]-->[1, 1, 512]
        
        # pred_embeddings为torch.Size([1, 512])
        # output_ids = output_ids[0][output_ids[0] != -200]
        # text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        # text_output = text_output.replace("\n", "").replace("  ", " ")
        # print("text_output: ", text_output)
        # print(pred_embeddings.shape)
        
        # lang_embeddings = model.encode_instruction(lang_annotation, model.args.device)#[1, 16, 512]
        #torch.Size([1, 16, 512])
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
    #===============Initialization（CLIP/tokenizer）==============
    
    clip_image_processor, tokenizer, LCB_model = Model_init(args)
    LCB_model.resize_token_embeddings(len(tokenizer))
    state_path = args.llm_ckpt+'/pytorch_model.bin'
    state_dict = torch.load(state_path, map_location='cpu')
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('base_model.model.'):
            new_key = key.replace('base_model.model.', '')
        else:
            new_key = key
        new_state_dict[new_key] = value
    LCB_model.load_state_dict(new_state_dict, strict=False)

    LLM_model = LCB_model
    LLM_model.cuda()  
    print('Load LLM part successfully!')
    
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
    evaluate_policy(model, LLM_model, clip_image_processor, tokenizer, env,
                    conf_dir=Path(args.calvin_model_path) / "conf",
                    eval_log_dir=args.base_log_dir,
                    sequence_indices=sequence_indices,
                    save_video=args.save_video)

    results, sequence_inds = collect_results(args.base_log_dir)
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
        for i, v in enumerate(count_success(results))]) + "|"
    )
    print(f'Load {len(results)}/100 episodes...')
    print(str_results + "\n")

    del env
    gc.collect()

def Add_LoRA(model, tokenizer):
        # ===============Add Lora===============
    lora_r = 8
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    return model

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

def preprocess(x: torch.Tensor) -> torch.Tensor:
    # Normalize colors
    img_size = 224
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

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
                short_input_ids.append(torch.tensor(tokenizer(parts[0]).input_ids, dtype=torch.long))
    input_ids = torch.nn.utils.rnn.pad_sequence(short_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()  
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

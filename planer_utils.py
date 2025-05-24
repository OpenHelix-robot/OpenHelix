import argparse
import transformers
from transformers import CLIPImageProcessor
import cv2
import torch
import torch.nn.functional as F
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)
import time
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX)
import numpy as np
from peft import LoraConfig, get_peft_model
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from planer import LISAForCausalLM
from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--llava_dir", default="LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1/LLaVA-7B-Lightening-v1-1", type=str)
    parser.add_argument("--vision_tower", default="clip-vit-large-patch14", type=str)
    return parser.parse_args()

def Add_LoRA(model, tokenizer, lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_target_modules="q_proj,v_proj"):
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

        lora_alpha = lora_alpha
        lora_dropout = lora_dropout
        lora_target_modules = find_linear_layers(
            model, lora_target_modules.split(",")
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
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    return model

def Model_init(vision_tower, llava_dir, torch_dtype):
    
    clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    tokenizer = transformers.AutoTokenizer.from_pretrained(llava_dir, cache_dir=None, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("<ACT>")
    seg_token_idx = tokenizer("<ACT>", add_special_tokens=False).input_ids[0]

    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    
    model_args = {
        "out_dim": 512,
        "vision_tower": vision_tower,
        "use_mm_start_end": True,
    }
    
    model = LISAForCausalLM.from_pretrained(llava_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args).cuda()
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=0)
    model.get_model().initialize_lisa_modules(model.get_model().config)
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["text_hidden_fcs","pred_act_mlps","pred_pos_act","pred_rot_act", "pred_gripper_act"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
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

def input_processing(image_dir, conv_list, clip_image_processor, tokenizer):
    '''
    preprocess input (image/text)
    '''
    if isinstance(image_dir, str):
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_dir.permute(1,2,0)
    
    image_clip = clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(torch.bfloat16).cuda() 
    image = preprocess(image.permute(2, 0, 1).contiguous()).to(torch.bfloat16).cuda()                 
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conv_list]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    
    targets = input_ids.clone()
    from model.llava import conversation as conversation_lib
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v1']
    conv = conversation_lib.default_conversation.copy()
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conv_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)

            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
    truncate_len = tokenizer.model_max_length - 255

    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]
    
    return image_clip, image, input_ids, attention_masks, targets

def input_processing_batch(image_tensor, conv_list, clip_image_processor, tokenizer):
    '''
    preprocess input (image/text)
    '''
    image_clip_list = []
    image_list = []
    timeone = time.time()
    for i in range(image_tensor.shape[0]):
        
        image = image_tensor[i]
        image = image.cpu().numpy()
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        image = image.transpose(1, 2, 0) 
    
        image_clip = clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].cuda() # [3, 224, 224]
        image_clip_list.append(image_clip.unsqueeze(0))
        image = preprocess(image.permute(2, 0, 1).contiguous()).cuda()# [3, 224, 224]
        image_list.append(image.unsqueeze(0))
    timetwo = time.time()
    image = torch.cat(image_list, dim=0)
    image_clip = torch.cat(image_clip_list, dim=0)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conv_list]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda() 
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    
    
    targets = input_ids.clone()
    from model.llava import conversation as conversation_lib
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v1']
    conv = conversation_lib.default_conversation.copy()
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conv_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)

            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
    truncate_len = tokenizer.model_max_length - 255

    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]
    
    return image_clip, image, input_ids, attention_masks, targets

def tensor2img(image_tensor, save_path):
    tensor = image_tensor.to(dtype=torch.float32)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    to_pil = transforms.ToPILImage()
    jpg_image = to_pil(tensor)
    jpg_image.save(save_path)  # For JPG
    print("Image saved successfully!")

def input_processing_real_batch(image_tensor, conv_list, clip_image_processor, tokenizer):
    '''
    preprocess input (image/text)
    '''
    timeone = time.time()
    

    images = image_tensor.cpu().numpy()


    images = (np.clip(images, 0, 1) * 255).astype(np.uint8)

    images = images.transpose(0, 2, 3, 1)  # (batch_size, H, W, C)
    
    pil_images = [Image.fromarray(image) for image in images]

    image_clip_batch = clip_image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"]
    image_clip_batch = image_clip_batch.to(torch.bfloat16).cuda() # [batch, channels, height, width]


    image_batch = preprocess(image_tensor.contiguous()).to(torch.bfloat16).cuda()  # [batch, 3, 224, 224]


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
    input_ids = torch.nn.utils.rnn.pad_sequence(short_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda() 
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()
    targets[:,:] = IGNORE_INDEX

    truncate_len = tokenizer.model_max_length - 255
    
    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]

    return image_clip_batch, image_batch, input_ids, attention_masks, targets

    
    
    

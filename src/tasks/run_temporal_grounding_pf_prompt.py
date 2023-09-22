# Copyright (C) 2022 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0

import xxlimited
import torch
import os
import time
import random
import math
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling import ClipBertForGroundingPropFree, ClipBertForGroundingPropFreeTxt

from src.modeling.e2e_model import ClipBertVP, ClipBert

from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, get_rounded_percentage)
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer_downpad, setup_e2e_optimizer_pad, setup_e2e_optimizer, setup_e2e_optimizer_txtpad, setup_e2e_optimizer_txt

import numpy as np
from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list
from collections import defaultdict

from src.datasets.dataset_temporal_grounding_2d import ClipBertGroundingDataset, VideoGroundingCollator
import json
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from src.utils.load_save import _to_cuda

def Merge(dict1, dict2):
    return (dict2.update(dict1))


def mk_video_tg_datalist(raw_datalist, cfg):
    """
    Args:
        raw_datalist: list(dict)
        cfg:

    Returns:

    """
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
        LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

    if cfg.dataset == "charades":
        original_train_json = open("./data/txt_db/charades/train.json")
        original_train_data = json.load(original_train_json)

        original_test_json = open("./data/txt_db/charades/test.json")
        original_test_data = json.load(original_test_json)
    elif cfg.dataset == "anet":
        original_train_json = open("./data/txt_db/anet_retrieval/train.json")
        original_train_data = json.load(original_train_json)

        original_test_json = open("./data/txt_db/anet_retrieval/val_1.json")
        original_test_data = json.load(original_test_json)

    Merge(original_test_data, original_train_data)

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        for i in range(len(raw_d["caption_list"])):
            if cfg.dataset == 'charades':
                d = dict(
                    id=qid,
                    txt=raw_d["caption_list"][i],
                    time_stamp=raw_d["timestamps"][i],
                    vid_id=raw_d["clip_name"],
                    duration = original_train_data[raw_d["clip_name"]]
                )
            elif cfg.dataset == "anet":
                d = dict(
                id=qid,
                txt=raw_d["caption_list"][i],
                time_stamp=raw_d["timestamps"][i],
                vid_id=raw_d["clip_name"],
                duration = original_train_data[raw_d["clip_name"]]['duration']
                )
            qid += 1
            datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")
    return datalist


def mk_video_tg_dataloader(anno_path, lmdb_dir, cfg, tokenizer, is_train=True):
    """"""
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_video_tg_datalist(raw_datalist, cfg)
    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group= 1,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")

    frm_sampling_strategy = "uniform"

    dataset = ClipBertGroundingDataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        itm_neg_size=cfg.itm_neg_size,
        ensemble_n_clips=cfg.train_n_clips,
        random_sample_clips=cfg.random_sample_clips
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    vtg_collator = VideoGroundingCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vtg_collator.collate_batch)
    return dataloader


def mk_video_tg_eval_dataloader(anno_path, lmdb_dir, cfg, tokenizer, is_train=True):
    """"""
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_video_tg_datalist(raw_datalist, cfg)
    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group= 1,
        # max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")

    frm_sampling_strategy = "uniform"

    dataset = ClipBertGroundingDataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        itm_neg_size=cfg.itm_neg_size,
        ensemble_n_clips=cfg.train_n_clips,
        random_sample_clips=cfg.random_sample_clips
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    vtg_collator = VideoGroundingCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vtg_collator.collate_batch)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loader = mk_video_tg_dataloader(
        anno_path=cfg.train_datasets[0].txt,
        lmdb_dir=cfg.train_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )
    val_loader = mk_video_tg_eval_dataloader(
        anno_path=cfg.val_datasets[0].txt,
        lmdb_dir=cfg.val_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=False
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loader = PrefetchLoader(val_loader, img_norm)
    return train_loader, val_loader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = [
        "num_labels", "classifier", "cls_hidden_scale",
        "loss_type", "margin",
    ]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    
    if cfg.add_txt_prompt:
        transformer_model_cls = ClipBertForGroundingPropFreeTxt
    else:
        transformer_model_cls = ClipBertForGroundingPropFree

    if cfg.add_txt_prompt:
        if cfg.add_vis_prompt:
            LOGGER.info("Setup e2e model with Txt Prompts and Visual Prompts!!")
        else:
            LOGGER.info("Setup e2e model with Txt Prompts ONLY !!")
    else:
        if cfg.add_vis_prompt:
            LOGGER.info("Setup e2e model with Visual Prompts ONLY !!")
        else:
            LOGGER.info("Setup e2e base model without ANY Prompts !!")

    if cfg.add_vis_prompt:
        model = ClipBertVP(
        model_cfg, cfg, input_format=cfg.img_input_format,
        detectron2_model_cfg=cfg.detectron2_model_cfg,
        transformer_cls=transformer_model_cls)
    else:
        model = ClipBert(
        model_cfg, input_format=cfg.img_input_format,
        detectron2_model_cfg=cfg.detectron2_model_cfg,
        transformer_cls=transformer_model_cls)

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    
    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading cnn weights from {cfg.detectron2_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            cnn_weights_path=cfg.detectron2_weights_path,
            bert_weights_path=cfg.bert_weights_path)

    if cfg.tuning_type == 'whole':
        for p in model.parameters():
                p.requires_grad = True
        LOGGER.info("The whole network would be UPDATED!! ")
    elif cfg.tuning_type == 'onlytxt':
        for p in model.parameters():
                p.requires_grad = False
        LOGGER.info("The whole network has been FROZEN! ")
        
        for n, p in model.named_parameters():
            if 'text_prompt' in n:
                p.requires_grad = True 
                print(n)
                LOGGER.info("------------ONLY Txt Prompts would be trained!!!------------------- ")
    
    elif cfg.tuning_type == 'txtpad':
        for p in model.parameters():
                p.requires_grad = False
        LOGGER.info("The whole network has been FROZEN! ")

        for n, p in model.named_parameters():
            
            if 'tp.pad_down' in n:
                p.requires_grad = True 
                print(n)
                LOGGER.info("------------Down Prompter would be trained!!!------------------- ")
            
            if cfg.vp_type == 'pad' or cfg.vp_type == 'framepad':
                if 'tp.pad_up' in n:
                    p.requires_grad = True 
                    print(n)
                    LOGGER.info("------------Up Prompter would be trained!!!------------------- ")
                
                if 'tp.pad_left' in n:
                    p.requires_grad = True 
                    print(n)
                    LOGGER.info("------------Left Prompter would be trained!!!------------------- ")
                
                if 'tp.pad_right' in n:
                    p.requires_grad = True 
                    print(n)
                    LOGGER.info("------------Right Prompter would be trained!!!------------------- ")

            if 'text_prompt' in n:
                    p.requires_grad = True 
                    print(n)
                    LOGGER.info("------------Txt Prompts would be trained!!!------------------- ")
    
    elif cfg.tuning_type == 'onlypad':
        for p in model.parameters():
                p.requires_grad = False
        LOGGER.info("The whole network has been FROZEN! ")

        if cfg.vp_apply == 'remove':
            LOGGER.info("No Prompts. Just REMOVE!! ")
        else:
            for n, p in model.named_parameters():
                
                if 'tp.pad_down' in n:
                    p.requires_grad = True 
                    print(n)
                    LOGGER.info("------------Down Prompter would be trained!!!------------------- ")
                
                if cfg.vp_type == 'pad' or cfg.vp_type == 'framepad':
                    if 'tp.pad_up' in n:
                        p.requires_grad = True 
                        print(n)
                        LOGGER.info("------------Up Prompter would be trained!!!------------------- ")
                    
                    if 'tp.pad_left' in n:
                        p.requires_grad = True 
                        print(n)
                        LOGGER.info("------------Left Prompter would be trained!!!------------------- ")
                    
                    if 'tp.pad_right' in n:
                        p.requires_grad = True 
                        print(n)
                        LOGGER.info("------------Right Prompter would be trained!!!------------------- ")


    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()
        LOGGER.info("------------CNN is FROZEN!!!------------------- ")

    if cfg.freeze_txtp:
        for n, p in model.named_parameters():
            if 'text_prompt' in n:
                p.requires_grad = False 
                print(n)
                LOGGER.info("------------Txt Prompts are FROZEN!!!------------------- ")

    if cfg.freeze_pad:
        for n, p in model.named_parameters():
            
            if 'tp.pad_down' in n:
                p.requires_grad = False 
                print(n)
                LOGGER.info("------------Down Prompter is FROZEN!!!------------------- ")
            
            if cfg.vp_type == 'pad' or cfg.vp_type == 'framepad':
                if 'tp.pad_up' in n:
                    p.requires_grad = False 
                    print(n)
                    LOGGER.info("------------Up Prompter is FROZEN!!!------------------- ")
                
                if 'tp.pad_left' in n:
                    p.requires_grad = False 
                    print(n)
                    LOGGER.info("------------Left Prompter is FROZEN!!!------------------- ")
                
                if 'tp.pad_right' in n:
                    p.requires_grad = False 
                    print(n)
                    LOGGER.info("------------Right Prompter is FROZEN!!!------------------- ")
        
        
    model.to(device)

    LOGGER.info("Setup model done!")
    return model


def forward_step(model, batch, cfg):
    """shared for training and validation"""
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def validate(model, eval_loader, cfg, train_global_step, eval_filepath):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()

    loss = 0.
    n_ex = 0
    n_corrects = 0
    st = time.time()
    debug_step = 5

    total_loss = 0
    total_iou_loss = 0
    total_dc_loss = 0
    total_duration_loss = 0

    total_val_count = 0
    count_iou3 = 0
    count_iou5 = 0
    count_iou7 = 0
    count_iou9 = 0
    iou_sum = 0
    for val_step, batch in enumerate(eval_loader):
        # forward pass
        del batch["caption_ids"]

        # could be 1, where only a single clip is used
        num_clips = cfg.train_n_clips
        num_frm = cfg.num_frm

        outputs = forward_step(model, batch, cfg)

        iou_scores = iou_eval(
            torch.mul(outputs["logits"], batch["duration"].view(batch["visual_inputs"].shape[0], 1)),
            batch["timestamp"])

        loss, iou_loss, dc_loss, duration_loss = dd_iou_loss(
            torch.mul(outputs["logits"], batch["duration"].view(batch["visual_inputs"].shape[0], 1)),
            batch["timestamp"], batch["duration"].view(batch["visual_inputs"].shape[0], 1), cfg.alpha, cfg.beta)
        
        loss = loss.mean()
        iou_loss = iou_loss.mean()
        dc_loss = dc_loss.mean()
        duration_loss = duration_loss.mean()

        total_val_count += batch["visual_inputs"].shape[0]

        count_iou3 += torch.numel(iou_scores[iou_scores>0.3])
        count_iou5 += torch.numel(iou_scores[iou_scores>0.5])
        count_iou7 += torch.numel(iou_scores[iou_scores>0.7])
        count_iou9 += torch.numel(iou_scores[iou_scores>0.9])
        iou_sum += torch.sum(iou_scores)

        total_loss += loss * batch["visual_inputs"].shape[0]
        total_iou_loss += iou_loss * batch["visual_inputs"].shape[0]
        total_dc_loss += dc_loss * batch["visual_inputs"].shape[0]
        total_duration_loss += duration_loss * batch["visual_inputs"].shape[0]

        if cfg.debug and val_step >= debug_step:
            break
    
    iou3 = (count_iou3 / total_val_count) * 100
    iou5 = (count_iou5 / total_val_count) * 100
    iou7 = (count_iou7 / total_val_count) * 100
    iou9 = (count_iou9 / total_val_count) * 100
    iou_mean = (iou_sum / total_val_count)* 100

    mean_loss = total_loss / total_val_count
    mean_iou_loss = total_iou_loss / total_val_count
    mean_dc_loss = total_dc_loss / total_val_count
    mean_duration_loss = total_duration_loss / total_val_count

    model.train()

    if hvd.rank() == 0:

        LOGGER.info(f'IoU@0.3 = {iou3}')
        LOGGER.info(f'IoU@0.5 = {iou5}')
        LOGGER.info(f'IoU@0.7 = {iou7}')
        LOGGER.info(f'IoU@0.9 = {iou9}')
        LOGGER.info(f'mIoU = {iou_mean}')

        LOGGER.info(f'total loss = {mean_loss}')
        LOGGER.info(f'IoU loss = {mean_iou_loss}')
        LOGGER.info(f'dc loss = {mean_dc_loss}')
        LOGGER.info(f'duration loss = {mean_duration_loss}')

def iou_eval(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0][:].float(), gt[1][:].float()

    inter = torch.min(e, end) - torch.max(s, start)
    union = torch.max(e, end) - torch.min(s, start)
    iou = inter.clamp(min=0) / union
    return iou


def dis_iou_loss(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0][:].float(), gt[1][:].float()

    mid_c = torch.div(torch.add(start, end), 2.0)
    mid_g = torch.div(torch.add(s, e), 2.0)

    inter = torch.min(e, end) - torch.max(s, start)
    union = torch.max(e, end) - torch.min(s, start)
    iou = inter.clamp(min=0) / union

    dis = torch.div(
        torch.square(torch.max(mid_g, mid_c) - torch.min(mid_g, mid_c)), 
        torch.square(torch.add(torch.sub(end, start), torch.sub(e, s)))
    )

    loss = 1-iou + dis
    
    return loss

def dd_iou_loss(candidates, gt, duration, alpha, beta):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0][:].float(), gt[1][:].float()

    mid_c = torch.div(torch.add(start, end), 2.0)
    mid_g = torch.div(torch.add(s, e), 2.0)

    inter = torch.min(e, end) - torch.max(s, start)
    union = torch.max(e, end) - torch.min(s, start)
    iou = inter.clamp(min=0) / union

    duration_es = torch.sub(end, start)
    duration_gt = torch.sub(e, s)

    # Losses
    iou_loss = 1 - iou

    d_c = torch.div(torch.max(mid_g, mid_c) - torch.min(mid_g, mid_c), duration)
    d_c = d_c.clamp(min=0.2)

    duration_diff = torch.square(torch.div(torch.sub(duration_es, duration_gt), duration))
    duration_diff = duration_diff.clamp(min=0.4)

    loss = iou_loss + alpha * d_c + beta * duration_diff

    return loss, iou_loss, d_c, duration_diff


def start_training(cfg):
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    cfg.n_gpu = n_gpu
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    LOGGER.info("--------Start Model Weight Loading --------")
    # cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.eval_dir, f"ckpt/model_step_{cfg.inference_model_step}.pt")
    if exists(e2e_weights_path):
        cfg.e2e_weights_path = e2e_weights_path
    else:
        raise ValueError(f"Non-valid checkpoint dir")
    model = setup_model(cfg, device=device)
    model.train()


    if cfg.add_txt_prompt:
        if cfg.add_vis_prompt:
            optimizer = setup_e2e_optimizer_txtpad(model, cfg)
        else:
            optimizer = setup_e2e_optimizer_txt(model, cfg)
    else:
        if cfg.vp_type == 'downpad' or cfg.vp_type == 'framedownpad':
            optimizer = setup_e2e_optimizer_downpad(model, cfg)
        elif cfg.vp_type == 'pad' or cfg.vp_type == 'framepad':
            optimizer = setup_e2e_optimizer_pad(model, cfg)
        elif cfg.vp_apply == 'remove':
            optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level='O2',
        keep_batchnorm_fp32=True)

    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    train_loader, eval_loader = setup_dataloaders(cfg, tokenizer)

    if cfg.do_check:
        global_step = 0
        cfg.data_ratio = 1.
        LOGGER.info(cfg)
        LOGGER.info("Starting inference...")
        LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
        LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

        LOGGER.info(f'Start validation')
        validate(
            model, eval_loader, cfg, global_step,
            eval_filepath=cfg.inference_txt_db)
        return

    # compute the number of steps and update cfg
    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))

    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        path = join(
            cfg.output_dir, 'log', "detectron2_model_cfg.yaml")
        with open(path, "w") as f:
            f.write(model.cnn.config_file)
        LOGGER.info("Saving training done...")
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 3
    running_loss = RunningMeter('train_loss')

    N = cfg.num_prop

    for step, batch in enumerate(InfiniteIterator(train_loader)):
        # forward pass
        del batch["caption_ids"]

        # Batch shape torch.Size([6, 64, 3, 448, 448])

        outputs = forward_step(model, batch, cfg)

        loss,_,_,_ = dd_iou_loss(
            torch.mul(outputs["logits"], batch["duration"].view(batch["visual_inputs"].shape[0], 1)),
            batch["timestamp"], batch["duration"].view(batch["visual_inputs"].shape[0], 1), cfg.alpha, cfg.beta)

        loss = loss.mean()

        running_loss(loss.item())
        # backward pass
        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            n_epoch = int(1. * total_train_batch_size * global_step
                            / total_n_examples)
            # learning rate scheduling transformer
            lr_this_step_transformer = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            # learning rate scheduling cnn
            lr_this_step_cnn = get_lr_sched(
                global_step, cfg.cnn_lr_decay, cfg.cnn_learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.cnn_step_decay_epochs,
                multi_step_epoch=n_epoch)

            # learning rate scheduling tp
            lr_this_step_tp = get_lr_sched(
                global_step, cfg.tp_lr_decay, cfg.tp_learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.tp_step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            #assert len(optimizer.param_groups) == 8
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                        cfg.transformer_lr_mul * lr_this_step_transformer)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step_transformer
                elif pg_n in [4, 5]:
                    param_group['lr'] = (
                        cfg.cnn_lr_mul * lr_this_step_cnn)
                elif pg_n in [6, 7]:
                    param_group['lr'] = lr_this_step_cnn
                else:   # [8, 9, 10, 11]
                    param_group['lr'] =  (
                        cfg.tp_lr_mul * lr_this_step_tp)
            
            TB_LOGGER.add_scalar(
                "train/lr_transformer", lr_this_step_transformer,
                global_step)
            TB_LOGGER.add_scalar(
                "train/lr_cnn", lr_this_step_cnn, global_step)

            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer),
                    cfg.grad_norm)
                TB_LOGGER.add_scalar(
                    "train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()
            pbar.update(1)

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                if cfg.skip_val:
                    model_saver.save(step=global_step, model=model)
                else:
                    LOGGER.info(f'Step {global_step}: start validation')
                    validate(
                        model, eval_loader, cfg, global_step,
                        eval_filepath=cfg.val_datasets[0].txt)
                    model_saver.save(step=global_step, model=model)



        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break



def start_inference(cfg):
    set_random_seed(cfg.seed)
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True

    inference_res_dir = join(
        cfg.output_dir,
        f"results_{os.path.splitext(os.path.basename(cfg.inference_txt_db))[0]}/"
        f"step_{cfg.inference_model_step}_{cfg.inference_n_clips}_{cfg.score_agg_func}"
    )

    if hvd.rank() == 0:
        os.makedirs(inference_res_dir, exist_ok=True)
        save_json(cfg, join(inference_res_dir, "raw_args.json"),
                  save_pretty=True)

    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    # overwrite cfg with stored_cfg,
    # but skip keys containing the keyword 'inference'
    stored_cfg_path = join(cfg.output_dir, "log/args.json")
    stored_cfg = edict(load_json(stored_cfg_path))
    for k, v in cfg.items():
        if k in stored_cfg and "inference" not in k and "output_dir" not in k:
            setattr(cfg, k, stored_cfg[k])

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.output_dir, f"ckpt/model_step_{cfg.inference_model_step}.pt")
    if exists(e2e_weights_path):
        cfg.e2e_weights_path = e2e_weights_path
    else:
        cfg.bert_weights_path = join(
            f"{cfg.output_dir}/ckpt",
            f"transformer_step_{cfg.inference_model_step}.pt")
        cfg.cnn_weights_path = join(
            cfg.output_dir, f"ckpt/cnn_step_{cfg.inference_model_step}.pt")
    model = setup_model(cfg, device=device)
    model.eval()

    # FIXME separate scaling for each loss
    model = amp.initialize(
        model, enabled=cfg.fp16, opt_level='O2')

    global_step = 0
    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    cfg.data_ratio = 1.

    val_loader = mk_video_tg_eval_dataloader(
        anno_path=cfg.inference_txt_db,
        lmdb_dir=cfg.inference_img_db,
        cfg=cfg, tokenizer=tokenizer,
    )

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    ret_results, ret_scores = inference_retrieval(
        model, val_loader, cfg.inference_txt_db, cfg)

    if hvd.rank() == 0:
        save_json(cfg, join(inference_res_dir, "merged_args.json"),
                  save_pretty=True)
        save_json(ret_results, join(inference_res_dir, "results.json"),
                  save_pretty=True)
        save_json(ret_scores, join(inference_res_dir, "scores.json"),
                  save_pretty=True)


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    input_cfg = shared_configs.get_video_retrieval_args()
    if input_cfg.do_inference:
        start_inference(input_cfg)
    else:
        start_training(input_cfg)

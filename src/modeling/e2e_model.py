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

from imaplib import Time2Internaldate
from src.modeling.modeling import ClipBertForPreTraining
from src.modeling.grid_feat import GridFeatBackbone
from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
import torch


class ClipBert(nn.Module):
    def __init__(self, config, input_format="BGR",
                 detectron2_model_cfg=None,
                 transformer_cls=ClipBertForPreTraining):
        super(ClipBert, self).__init__()
        self.config = config
        self.detectron2_model_cfg = detectron2_model_cfg
        assert detectron2_model_cfg is not None
        cnn_cls = GridFeatBackbone
        print(f"cnn_cls {cnn_cls}")
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)
        self.transformer = transformer_cls(config)

    def forward(self, batch):
        # used to make visual feature copies
        # repeat_counts = batch["n_examples_list"]
        # repeat_counts = 1
        del batch["n_examples_list"]
        del batch["duration"]
        del batch["timestamp"]
        visual_features = self.cnn(batch["visual_inputs"])
        # batch["visual_inputs"] = repeat_tensor_rows(
        #     visual_features, repeat_counts)
        batch["visual_inputs"] = visual_features
        # if self.retrieval:
        #     batch["sample_size"] = len(repeat_counts)  # batch size
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False

class ResNet50(nn.Module):
    def __init__(self, config, input_format="BGR",
                 detectron2_model_cfg=None):
        super(ResNet50, self).__init__()
        self.config = config
        self.detectron2_model_cfg = detectron2_model_cfg
        assert detectron2_model_cfg is not None
        cnn_cls = GridFeatBackbone
        print(f"cnn_cls {cnn_cls}")
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)

    def forward(self, batch):
        # used to make visual feature copies
        # repeat_counts = batch["n_examples_list"]
        # repeat_counts = 1
        # del batch["n_examples_list"]
        # del batch["duration"]
        # del batch["timestamp"]
        outputs = self.cnn(batch["visual_inputs"])
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False

class FixedPatchPrompter(nn.Module):
    def __init__(self, image_size, prompt_size):
        super(FixedPatchPrompter, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :, :self.psize, :self.psize] = self.patch

        return x + prompt

class PadPrompter(nn.Module):
    def __init__(self, image_size, prompt_size):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left.to(torch.float16), base.to(torch.float16), self.pad_right.to(torch.float16)], dim=4)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
        prompt = torch.cat(x.size(1) * [prompt], dim = 1)
        prompt = torch.cat(x.size(0) * [prompt])
        return prompt

class DownPadPrompter(nn.Module):
    def __init__(self, image_size, pad_size):
        super(DownPadPrompter, self).__init__()
        self.pad_size = pad_size
        self.image_size = image_size

        self.pad_down = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))

    def forward(self, x):
        prompt = torch.zeros([x.shape[0], x.shape[1], 3, self.image_size, self.image_size]).cuda()
        start_point = self.image_size- self.pad_size
        prompt[:, :, :, start_point:self.image_size, :] = self.pad_down
        return prompt

class FrameDownPadPrompter(nn.Module):
    def __init__(self, image_size, pad_size, frame_num):
        super(FrameDownPadPrompter, self).__init__()
        self.pad_size = pad_size
        self.image_size = image_size
        self.frame_num = frame_num

        self.pad_down = nn.Parameter(torch.randn([1, frame_num, 3, pad_size, image_size]))

    def forward(self, x):
        prompt = torch.zeros([x.shape[0], x.shape[1], 3, self.image_size, self.image_size]).cuda()
        start_point = self.image_size- self.pad_size
        prompt[:, :, :, start_point:self.image_size, :] = self.pad_down
        return prompt


class FramePadPrompter(nn.Module):
    def __init__(self, image_size, prompt_size, frame_num):
        super(FramePadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size
        self.frame_num = frame_num

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, frame_num, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, frame_num, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, frame_num, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, frame_num, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, self.frame_num, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left.to(torch.float16), base.to(torch.float16), self.pad_right.to(torch.float16)], dim=4)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
        # prompt = torch.cat(x.size(1) * [prompt], dim = 1)
        prompt = torch.cat(x.size(0) * [prompt])
        return prompt

class ClipBertVP(nn.Module):
    def __init__(self, config, cfg, input_format="BGR",
                 detectron2_model_cfg=None,
                 transformer_cls=ClipBertForPreTraining):
        super(ClipBertVP, self).__init__()
        self.config = config
        self.cfg = cfg
        self.detectron2_model_cfg = detectron2_model_cfg
        assert detectron2_model_cfg is not None
        cnn_cls = GridFeatBackbone
        print(f"cnn_cls {cnn_cls}")
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)
        self.transformer = transformer_cls(config)
        
        if cfg.vp_type == 'downpad':
            self.tp = DownPadPrompter(cfg.max_img_size, cfg.pad_size)
        elif cfg.vp_type == 'pad':
            self.tp = PadPrompter(cfg.max_img_size, cfg.pad_size)
        elif cfg.vp_type == 'framedownpad':
            self.tp = FrameDownPadPrompter(cfg.max_img_size, cfg.pad_size, cfg.num_frm)
        elif cfg.vp_type == 'framepad':
            self.tp = FramePadPrompter(cfg.max_img_size, cfg.pad_size, cfg.num_frm)

    def forward(self, batch):
        # used to make visual feature copies
        # repeat_counts = batch["n_examples_list"]
        # repeat_counts = 1

        del batch["n_examples_list"]
        del batch["duration"]
        del batch["timestamp"]

        if self.cfg.vp_apply == 'remove':
            remove = 1
        else:
            vp = self.tp (batch["visual_inputs"])


        if self.cfg.vp_apply == 'add':
            batch["visual_inputs"] = batch["visual_inputs"]  + vp

        elif self.cfg.vp_apply == 'remove':
            tp_mask = torch.ones([self.cfg.max_img_size, self.cfg.max_img_size]).cuda()
            start_point = self.cfg.pad_size
            end_point = self.cfg.max_img_size - self.cfg.pad_size
            
            if self.cfg.vp_type == 'downpad' or self.cfg.vp_type == 'framedownpad':
                tp_mask[end_point:self.cfg.max_img_size, :] = 0.0
            elif self.cfg.vp_type == 'pad':
                tp_mask[end_point:self.cfg.max_img_size, :] = 0.0
                tp_mask[:start_point, :] = 0.0
                tp_mask[: , end_point:self.cfg.max_img_size] = 0.0
                tp_mask[: , :start_point] = 0.0

            batch["visual_inputs"] = batch["visual_inputs"]*tp_mask

        elif self.cfg.vp_apply == 'replace':
            tp_mask = torch.ones([self.cfg.max_img_size, self.cfg.max_img_size]).cuda()
            start_point = self.cfg.pad_size
            end_point = self.cfg.max_img_size - self.cfg.pad_size
            
            if self.cfg.vp_type == 'downpad' or self.cfg.vp_type == 'framedownpad':
                tp_mask[end_point:self.cfg.max_img_size, :] = 0.0
            elif self.cfg.vp_type == 'pad':
                tp_mask[end_point:self.cfg.max_img_size, :] = 0.0
                tp_mask[:start_point, :] = 0.0
                tp_mask[: , end_point:self.cfg.max_img_size] = 0.0
                tp_mask[: , :start_point] = 0.0

            batch["visual_inputs"] = batch["visual_inputs"]*tp_mask  + vp
        
        batch["visual_inputs"] = batch["visual_inputs"].to(torch.float16).cuda()

        batch["visual_inputs"] = self.cnn(batch["visual_inputs"])
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None, tp_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

        if tp_weights_path:
            self.cnn.load_state_dict(tp_weights_path)
        

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False

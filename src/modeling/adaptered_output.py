import copy
import os
import sys
import logging
from accelerate.logging import get_logger
import itertools
import pdb
import time
from PIL import Image
from typing import List, Dict
from typing_extensions import OrderedDict


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ViltConfig, ViltProcessor, ViltModel
from transformers import BertTokenizerFast
from transformers import logging as transformers_logging
from src.modeling.models.adapter import Adapter, init_bert_weights
from src.modeling.continual_learner import EncoderWrapper, ContinualLearner
from src.modeling.models.vit import Attention
import loralib as lora

class Attention_lorad(nn.Module):
    def __init__(self, layer, dim):
        super().__init__()
        self.layer = layer
        self.query = lora.Linear(dim, dim, r=16)
        self.value = lora.Linear(dim, dim, r=16)

    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.layer.qkv(x).reshape(B, N, 3, self.layer.num_heads, C // self.layer.num_heads).permute(2, 0, 3, 1, 4)
        q_lora = self.query(x).reshape(B, N, self.layer.num_heads, C // self.layer.num_heads).permute(0, 2, 1, 3)
        v_lora = self.value(x).reshape(B, N, self.layer.num_heads, C // self.layer.num_heads).permute(0, 2, 1, 3)
        q, k, v = q_lora, qkv[1], v_lora  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.layer.scale
        attn = attn.softmax(dim=-1)
        attn = self.layer.attn_drop(attn)

        if register_hook:
            self.layer.save_attention_map(attn)
            attn.register_hook(self.layer.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.layer.proj(x)
        x = self.layer.proj_drop(x)
        return x

class Adaptered_BertOutput(nn.Module):
    def __init__(self, layer, adapter_config):
        super().__init__()
        self.layer = layer
        self.adapter = Adapter(**adapter_config, model_dim=768)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.layer.dense(hidden_states)
        hidden_states = self.layer.dropout(hidden_states)
        hidden_states = self.adapter.adapter_layer_forward_bert(hidden_states, input_tensor, self.layer.LayerNorm)
        return hidden_states

class Adaptered_ViltOutput(nn.Module):
    def __init__(self, layer, adapter_config) -> None:
        super().__init__()
        self.layer = layer
        self.adapter = Adapter(**adapter_config, model_dim=768)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer.dense(hidden_states)
        hidden_states = self.layer.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        hidden_states = self.adapter(hidden_states, hidden_states)
        return hidden_states
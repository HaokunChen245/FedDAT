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

from src.modeling.continual_learner import EncoderWrapper, ContinualLearner


class ViltForImageClassification(nn.Module):

    def __init__(self, encoder_dim: int, num_labels: int):
        '''
        Modified ViLT model for image classification tasks
        args:
        encoder - instance of ViltEncoderWrapper class
        encoder_dim - output dimension of vilt encoder
        num_labels - number of output labels for image classification task
        '''

        super().__init__()
        self.encoder_dim = encoder_dim
        self.clf_layer = nn.Sequential(
                            nn.Linear(encoder_dim, encoder_dim*2),
                            nn.LayerNorm(encoder_dim*2),
                            nn.GELU(),
                            nn.Linear(encoder_dim*2, num_labels)
                        )

    def forward(self, encoder, images: List, texts: List[str]) -> torch.FloatTensor:
        '''
        Does forward pass of image and text inputs through model, where texts are dummy texts

        Args:
        images - batch_size-sized list of num_images-sized list of PIL Image objects
        texts - list of dummy text strings
        '''
        encodings = encoder.process_inputs(images, texts)
        encoder_output = encoder(**encodings)

        output_logits = self.clf_layer(encoder_output)
        return output_logits


class ViltForSequenceClassification(nn.Module):

    def __init__(self, encoder_dim: int, num_labels: int):
        '''
        Modified ViLT model for text classification tasks

        Args:
        encoder_dim - output dimension of vilt encoder
        num_labels - number of output labels for text classification task
        '''

        super().__init__()
        self.encoder_dim = encoder_dim
        self.clf_layer = nn.Sequential(
                            nn.Linear(encoder_dim, encoder_dim*2),
                            nn.LayerNorm(encoder_dim*2),
                            nn.GELU(),
                            nn.Linear(encoder_dim*2, num_labels)
                        )

    def forward(self, encoder, images: List, texts: List[str]) -> torch.FloatTensor:
        '''
        Does forward pass of image and text inputs through model, where image is averaged image

        Args:
        images - batch_size-sized list of "average image"'sPIL Image objects
        texts - list of text strings
        '''

        encodings = encoder.process_inputs(images, texts)
        # expand to batch size
        bs = len(encodings['input_ids'])
        encodings['pixel_values'] = encodings['pixel_values'].expand([bs, *encodings['pixel_values'].shape[1:]])
        encodings['pixel_mask'] = encodings['pixel_mask'].expand([bs, *encodings['pixel_mask'].shape[1:]])
        encoder_output = encoder(**encodings)
        output_logits = self.clf_layer(encoder_output)
        return output_logits


class ViltForMultipleChoice(nn.Module):

    def __init__(self, encoder_dim: int, num_labels: int):
        '''
        Modified ViLT model for text multiple-choice tasks
        Args:
        encoder_dim - output dimension of vilt encoder
        num_labels - number of choices for multi-choice task
        '''
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_labels = num_labels
        self.clf_layer = nn.Sequential(
                            nn.Dropout(0.1),
                            nn.Linear(encoder_dim, 1)
                        )

    def forward(self, encoder, images, texts):
        encodings = encoder.process_inputs(images, texts)
        # unflat_input_ids = encodings['input_ids'].view(self.num_labels, 32, -1).transpose(0, 1)
        bs = len(encodings['input_ids'])
        encodings['pixel_values'] = encodings['pixel_values'].expand([bs, *encodings['pixel_values'].shape[1:]])
        encodings['pixel_mask'] = encodings['pixel_mask'].expand([bs, *encodings['pixel_mask'].shape[1:]])
        encoder_output = encoder(**encodings)
        reshape_output = encoder_output.view(self.num_labels, -1, self.encoder_dim).transpose(0, 1).contiguous()

        output_logits = self.clf_layer(reshape_output).squeeze()
        return output_logits
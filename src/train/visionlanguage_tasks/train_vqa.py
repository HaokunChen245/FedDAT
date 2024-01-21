import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil
import pickle as pkl
import copy
import pdb
from tqdm import tqdm
from typing import List, Dict, Tuple

sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from src.data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from src.data.visionlanguage_datasets.vqa_dataset import build_vqa_dataloader
from src.train.visionlanguage_tasks.task_trainer import TaskTrainer
from src.utils.wandb import wandb_logger

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class VQATrainer(TaskTrainer):

    def __init__(self,
                 logger,
                 args: argparse.Namespace,
                 task_configs: Dict,
                 model_config: Dict,
                 device: torch.device,
                 task_key,
                 task_output_dir,
                 accelerator):

        '''
        Initializes a Trainer that handles training of a model on the VCR task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        '''

        super().__init__()

        self.args = args
        self.local_epochs = args.local_epochs
        self.device = device
        self.accelerator = accelerator
        self.task_output_dir = task_output_dir
        self.task_key = task_key

        self.vqa_config = task_configs['vqa']
        self.data_dir = os.path.join(args.climb_data_dir, self.vqa_config['data_dir'])

        # Model-specific stuff
        self.visual_input_type = model_config['visual_input_type']
        self.batch2inputs_converter = model_config['batch2inputs_converter']

        # Load COCO Images dataset for image data backbone
        images_source = self.vqa_config['images_source']
        mscoco_config = task_configs[images_source]
        self.images_dataset = MSCOCOImagesDataset(coco_dir=os.path.join(args.climb_data_dir, mscoco_config['data_dir']),
                                                  visual_input_type=args.visual_input_type)

        # Create dataloaders for training and validation
        self.vqa_train_dataloader = build_vqa_dataloader(args=args,
                                                    data_dir=self.data_dir,
                                                    images_dataset=self.images_dataset,
                                                    split='train',
                                                    visual_input_type=self.visual_input_type)

        self.vqa_val_dataloader = build_vqa_dataloader(args=args,
                                                  data_dir=self.data_dir,
                                                  images_dataset=self.images_dataset,
                                                  split='val',
                                                  visual_input_type=self.visual_input_type)

        # Training hyperparameters
        self.num_epochs = self.vqa_config['num_epochs']
        self.lr = self.vqa_config['lr']
        self.adam_epsilon = self.vqa_config['adam_epsilon']
        self.weight_decay = self.vqa_config['weight_decay']
        self.hparams = {
                        'lr': self.lr,
                        'weight_decay': self.weight_decay,
                        'adam_epsilon': self.adam_epsilon,
        }

        self.loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        self.vqa_train_dataloader.dataset.convert_to_low_shot(low_shot_percentage=0.05)
        self.vqa_val_dataloader.dataset.convert_to_low_shot(low_shot_percentage=0.05)
        self.max_steps = len(self.vqa_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code

    def compute_score_with_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        Given logits for each answer in VQA classification, selects answer with max logit and returns VQA-score for that answer
        logits: logits for each answer - size=(batch_size, num_answers)
        labels: label for each answer in {0, 0.3, 0.6, 1} (batch_size, num_answers)

        Returns:
        scores: score of predicted answer (batch_size, num_answers)
        '''

        logits = torch.max(logits, 1)[1].data # argmax
        one_hots = torch.zeros(*labels.size()).to(self.device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)
        return scores

    def get_train_dataloader(self):
        return self.vqa_train_dataloader

    def get_collate_fn(self):
        return self.vqa_train_dataloader.collate_fn

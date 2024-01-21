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

from src.data.visionlanguage_datasets.vcr_dataset import build_vcr_dataloader
from src.train.visionlanguage_tasks.task_trainer import TaskTrainer
from src.utils.wandb import wandb_logger

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


class VCRTrainer(TaskTrainer):

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

        self.vcr_config = task_configs['vcr']
        self.data_dir = os.path.join(args.climb_data_dir, self.vcr_config['data_dir'])
        self.task_type = self.vcr_config['task_type']

        # Model-specific stuff
        self.visual_input_type = model_config['visual_input_type']
        self.batch2inputs_converter = model_config['batch2inputs_converter']

        # Create dataloaders for training and validation
        self.vcr_train_dataloader = build_vcr_dataloader(args=args,
                                                data_dir=self.data_dir,
                                                split='train',
                                                task_type=self.task_type,
                                                visual_input_type=self.visual_input_type)

        self.vcr_val_dataloader = build_vcr_dataloader(args=args,
                                                data_dir=self.data_dir,
                                                split='val',
                                                task_type=self.task_type,
                                                visual_input_type=self.visual_input_type)

        # Training hyperparameters
        self.num_epochs = self.vcr_config['num_epochs']
        self.lr = self.vcr_config['lr']
        self.adam_epsilon = self.vcr_config['adam_epsilon']
        self.weight_decay = self.vcr_config['weight_decay']
        self.loss_criterion = nn.CrossEntropyLoss()

        self.vcr_train_dataloader.dataset.convert_to_low_shot(low_shot_percentage=0.05)
        self.vcr_val_dataloader.dataset.convert_to_low_shot(low_shot_percentage=0.05)
        self.max_steps = len(self.vcr_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code
        self.hparams = {
                        'lr': self.lr,
                        'weight_decay': self.weight_decay,
                        'adam_epsilon': self.adam_epsilon,
        }

    def get_train_dataloader(self):
        return self.vcr_train_dataloader

    def get_collate_fn(self):
        return self.vcr_train_dataloader.collate_fn
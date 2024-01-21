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

from src.data.image_datasets.flickr30kimages_dataset import Flickr30KImagesDataset
from src.data.visionlanguage_datasets.snli_ve_dataset import build_snli_ve_dataloader
from src.train.visionlanguage_tasks.task_trainer import TaskTrainer
from src.utils.wandb import wandb_logger

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class SNLIVETrainer(TaskTrainer):

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

        self.snli_ve_config = task_configs['snli-ve']
        self.data_dir = os.path.join(args.climb_data_dir, self.snli_ve_config['data_dir'])

        # Model-specific stuff
        self.visual_input_type = model_config['visual_input_type']
        self.batch2inputs_converter = model_config['batch2inputs_converter']

        # Load Flickr30K Images dataset for image data backbone
        images_source = self.snli_ve_config['images_source']
        flickr30k_config = task_configs[images_source]
        images_dataset = Flickr30KImagesDataset(os.path.join(args.climb_data_dir, flickr30k_config['data_dir']),
                         visual_input_type=self.visual_input_type)

        # Create dataloaders for training and validation
        self.snli_ve_train_dataloader = build_snli_ve_dataloader(args=args,
                                                                 data_dir=self.data_dir,
                                                                 images_dataset=images_dataset,
                                                                 split='train',
                                                                 visual_input_type=self.visual_input_type)

        self.snli_ve_dev_dataloader = build_snli_ve_dataloader(args=args,
                                                               data_dir=self.data_dir,
                                                               images_dataset=images_dataset,
                                                               split='dev',
                                                               visual_input_type=self.visual_input_type)

        # Training hyperparameters
        self.num_epochs = self.snli_ve_config['num_epochs']
        self.lr = self.snli_ve_config['lr']
        self.adam_epsilon = self.snli_ve_config['adam_epsilon']
        self.weight_decay = self.snli_ve_config['weight_decay']
        self.loss_criterion = nn.CrossEntropyLoss()

        self.snli_ve_train_dataloader.dataset.convert_to_low_shot(num_shots_per_class=2048)
        self.snli_ve_dev_dataloader.dataset.convert_to_low_shot(num_shots_per_class=256)

        self.max_steps = len(self.snli_ve_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1 # TODO remove hard code
        self.hparams = {
                        'lr': self.lr,
                        'weight_decay': self.weight_decay,
                        'adam_epsilon': self.adam_epsilon,
        }

    def get_train_dataloader(self):
        return self.snli_ve_train_dataloader

    def get_collate_fn(self):
        return self.snli_ve_train_dataloader.collate_fn
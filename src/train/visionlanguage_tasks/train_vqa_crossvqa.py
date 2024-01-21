import copy
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
import pdb
from tqdm import tqdm
from typing import List, Dict
import torch.nn.functional as F

from src.modeling.continual_learner import ContinualLearner

sys.path.insert(0, ".")

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from src.data.image_datasets.vgimages_dataset import VGImagesDataset
from src.data.image_datasets.vizwizimages_dataset import vizwizImagesDataset
from src.data.visionlanguage_datasets.vqa_dataset_crossvqa import (
    build_vqa_vilt_dataloader,
    build_vqa_albef_dataloader,
)
from src.train.visionlanguage_tasks.task_trainer import TaskTrainer
from src.utils.wandb import wandb_logger
from src.modeling.models.tokenization_bert import BertTokenizer
from src.utils.seed_utils import set_seed


class VQATrainerCross(TaskTrainer):
    def __init__(
        self,
        logger,
        args: argparse.Namespace,
        task_configs: Dict,
        model_config: Dict,
        device: torch.device,
        task_key: str,
        task_output_dir=None,
        client_id=-1,
        accelerator=None,
    ):

        """
        Initializes a Trainer that handles training of a model on the VQA task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        """

        super().__init__()
        self.accelerator = accelerator
        self.device = self.accelerator.device
        # make sure different process gets different seed
        set_seed(args.seed + self.accelerator.process_index)
        self.logger = logger

        # Create W&B experiment
        if args.do_wandb_logging:
            self.logger.info(
                "W&B project: {}, experiment: {}".format(
                    "CARVEN", task_output_dir.split("/")[-3]
                )
            )
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers(project_name="CARVEN")
                self.accelerator.trackers[0].run.name = (
                    task_output_dir.split("/")[-4]
                    + "/"
                    + task_output_dir.split("/")[-3]
                    + "/"
                    + task_output_dir.split("/")[-1]
                )

        self.args = args
        self.local_epochs = args.local_epochs
        self.task_key = task_key
        self.task_output_dir = task_output_dir

        self.vqa_config = task_configs[self.task_key]  # in task_configs.py
        self.batch2inputs_converter = model_config["batch2inputs_converter"]

        # Model-specific stuff
        if "vilt" in args.encoder_name:
            self.model_name = "vilt"
            self.data_dir = os.path.join(
                args.climb_data_dir, self.vqa_config["data_dir"]
            )  # vqa_abstract
            self.visual_input_type = model_config["visual_input_type"]  # pil_image

            # Create dataloaders for training and validation
            # Load COCO Images dataset for image data backbone
            images_source = self.vqa_config["images_source"]
            if task_key=='gqa' or "clove" in task_key:
                self.images_dataset = VGImagesDataset(
                    coco_dir=args.climb_data_dir,
                    data_dir=None,
                    visual_input_type=args.visual_input_type,
                    task_key=self.task_key,
                )
            elif task_key=='vizwiz':
                self.images_dataset = vizwizImagesDataset(
                    coco_dir=args.climb_data_dir,
                    data_dir=None,
                    visual_input_type=args.visual_input_type,
                    task_key=self.task_key,
                )
            else:
                mscoco_config = task_configs[images_source]
                from src.data.image_datasets.cocoimages_dataset_crossvqas import MSCOCOImagesDataset
                self.images_dataset = MSCOCOImagesDataset(
                    coco_dir=args.climb_data_dir,
                    data_dir=mscoco_config["data_dir"],
                    visual_input_type=args.visual_input_type,
                    task_key=self.task_key,
                )

            self.vqa_train_dataloader = build_vqa_vilt_dataloader(
                logger=self.logger,
                args=args,
                data_dir=self.data_dir,
                images_dataset=self.images_dataset,
                split=self.args.splits[0],
                task_key=self.task_key,
                visual_input_type=self.visual_input_type,
                client_id=client_id,
            )

            self.vqa_val_dataloader = build_vqa_vilt_dataloader(
                logger=self.logger,
                args=args,
                data_dir=self.data_dir,
                images_dataset=self.images_dataset,
                split=self.args.splits[1],
                task_key=self.task_key,
                visual_input_type=self.visual_input_type,
                client_id=client_id,
            )

            self.vqa_test_dataloader = build_vqa_vilt_dataloader(
                logger=self.logger,
                args=args,
                data_dir=self.data_dir,
                images_dataset=self.images_dataset,
                split=self.args.splits[2],
                task_key=self.task_key,
                visual_input_type=self.visual_input_type,
                client_id=client_id,
            )
        else:
            self.model_name = "albef"
            self.data_dir = os.path.join(
                args.climb_data_dir, self.vqa_config["data_dir"]
            )  # vqa_abstract

            images_source = self.vqa_config["images_source"]
            if task_key=='gqa' or "clove" in task_key:
                self.images_dataset = VGImagesDataset(
                    coco_dir=args.climb_data_dir,
                    data_dir=None,
                    visual_input_type=args.visual_input_type,
                    task_key=self.task_key,
                )
            elif task_key=='vizwiz':
                self.images_dataset = vizwizImagesDataset(
                    coco_dir=args.climb_data_dir,
                    data_dir=None,
                    visual_input_type=args.visual_input_type,
                    task_key=self.task_key,
                )
            else:
                mscoco_config = task_configs[images_source]
                from src.data.image_datasets.cocoimages_dataset_crossvqas import MSCOCOImagesDataset
                self.images_dataset = MSCOCOImagesDataset(
                    coco_dir=args.climb_data_dir,
                    data_dir=mscoco_config["data_dir"],
                    visual_input_type=args.visual_input_type,
                    task_key=self.task_key,
                )
            self.vqa_train_dataloader = build_vqa_albef_dataloader(
                logger=self.logger,
                args=args,
                data_dir=self.data_dir,
                images_dataset=self.images_dataset,
                vqa_config=self.vqa_config,
                split=self.args.splits[0],
                task_key=self.task_key,
                client_id=client_id,
            )

            self.vqa_val_dataloader = build_vqa_albef_dataloader(
                logger=self.logger,
                args=args,
                data_dir=self.data_dir,
                images_dataset=self.images_dataset,
                vqa_config=self.vqa_config,
                split=self.args.splits[1],
                task_key=self.task_key,
                client_id=client_id,
            )

            self.vqa_test_dataloader = build_vqa_albef_dataloader(
                logger=self.logger,
                args=args,
                data_dir=self.data_dir,
                images_dataset=self.images_dataset,
                vqa_config=self.vqa_config,
                split=self.args.splits[2],
                task_key=self.task_key,
                client_id=client_id,
            )

        (
            self.vqa_train_dataloader,
            self.vqa_val_dataloader,
            self.vqa_test_dataloader,
        ) = self.accelerator.prepare(
            self.vqa_train_dataloader, self.vqa_val_dataloader, self.vqa_test_dataloader
        )

        # Training hyperparameters
        self.num_epochs = self.args.num_epochs
        self.lr = self.args.lr
        self.adam_epsilon = self.vqa_config["adam_epsilon"]
        self.weight_decay = self.vqa_config["weight_decay"]
        self.loss_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.max_steps = len(self.vqa_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1  # TODO remove hard code

    def compute_score_with_logits(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Given logits for each answer in VQA classification, selects answer with max logit and returns VQA-score for that answer
        logits: logits for each answer - size=(batch_size, num_answers)
        labels: label for each answer in {0, 0.3, 0.6, 1} (batch_size, num_answers)

        Returns:
        scores: score of predicted answer (batch_size, num_answers)
        """

        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size()).to(self.device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * labels
        return scores

    def get_train_dataloader(self):
        return self.vqa_train_dataloader

    def get_collate_fn(self):
        return self.vqa_train_dataloader.collate_fn

    def add_alpha(self, epoch, batch, step):
        if epoch > 0:  # alpha is for distill
            alpha = 0.4
        else:
            alpha = 0.4 * min(1, step / len(self.vqa_train_dataloader))
        batch.append(alpha)
        return batch

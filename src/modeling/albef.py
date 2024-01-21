import os
import sys
import logging
import itertools
import pdb
import time
from PIL import Image
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from src.modeling.models.albef_model import ALBEF
from src.modeling.models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast
from transformers import logging as transformers_logging

from src.modeling.continual_learner import EncoderWrapper, ContinualLearner


class ALBEFWrapper(EncoderWrapper):

    def __init__(self, albef: ALBEF, device: torch.device):
        """
        Wrapper around ALBEF model from huggingface library
        this is the class that gets saved during checkpointing for continual learning
        args:
        albef - instance of ALBEF class
        device - gpu/cuda
        """

        super().__init__()
        self.albef = albef
        self.device = device

        BERT_LOCAL_PATH = './models/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH, local_files_only=True)

    def forward(self, batch) -> torch.FloatTensor:
        """
        Does forward pass of input encodings through ALBEF

        Args:
        batch: List containing inputs ALBEF's forward pass

        Returns:
        loss
        """
        images = batch["images"].to(self.device, non_blocking=True)

        if batch["train"]:
            weights = batch["weights"].to(self.device, non_blocking=True)
            question_input = self.tokenizer(batch["questions"], padding="longest", truncation=True, max_length=25, return_tensors="pt", ).to(self.device)
            answer_input = self.tokenizer(batch["answers"], padding="longest", return_tensors="pt").to(self.device)
            loss, logits = self.albef(image=images, question=question_input, answer=answer_input, train=True,
                                      alpha=batch["alpha"], k=batch["n"], weights=weights,)
            return [loss, logits]
        else:
            question_input = self.tokenizer(batch["questions"], padding="longest", return_tensors="pt").to(self.device)
            answer_list = [answer + "[SEP]" for answer in batch["answer_list"]]
            answer_input = self.tokenizer(answer_list, padding="longest", return_tensors="pt").to(self.device)
            # todo: double-check: ALBEF pads to longest answer in the answer list, but we pad to 'max_length' because logits used in CL methods need to be of the same shape
            # abstract: 8; med: 49; pvqa: 51; art: 5; toronto: 6
            # answer_input: dict{inpput_ids: tensor, attention_mask: tensor, token_type_ids: tensor}, all tensors are of shape (num_answers, words)
            # if the answers from different dataset need to be of the same length, use following line
            # answer_input = self.tokenizer(answer_list, padding='max_length', return_tensors='pt', max_length=51).to(self.device)

            topk_ids, topk_probs = self.albef(image=images, question=question_input, answer=answer_input, train=False, k=batch["k"], )
            return [topk_ids, topk_probs]

    def freeze_all_weights(self):  # todo
        """
        Freeze all parameters in self.albef
        """

        for p in self.albef.parameters():
            p.requires_grad = False

    def freeze_bottom_k_layers(self, k: int):  # todo
        """
        Freeze embedding parameters and bottom K transformer layer parameters
        """

        assert k < len(self.albef.encoder.layer)
        for p in self.albef.embeddings.parameters():
            p.requires_grad = False
        for i in range(k):
            for p in self.albef.encoder.layer[i].parameters():
                p.requires_grad = False

    def freeze_encoder(self):  # todo
        raise NotImplementedError


class ALBEFContinualLearner(ContinualLearner):
    # I should be ALBEF now
    def __init__(self, ordered_cl_tasks: List[str], albef_model: ALBEFWrapper, task_configs: Dict):
        """
        The actual Continual Learning model

        arguments:
        ordered_cl_tasks - list of CL task keys that will be encountered by the ContinualLearner
        albef_model - instance of ALBEFEncoderWrapper class
        task_configs - dictionary which contains task-specific configurations/hparams for each task in ordered_cl_tasks, not used for now
        """

        super().__init__()
        self.albef_model = albef_model
        # self.ordered_cl_tasks = ordered_cl_tasks
        # self.task_configs = task_configs
        #
        # self.task_layer_dict = {} #
        # for task_key in ordered_cl_tasks:
        #         self.add_task_layer(task_key, task_configs[task_key])
        # self.task_layer = nn.ModuleDict(self.task_layer_dict)

    def set_active_lora(self):
        import loralib as lora
        for i in range(len(self.albef_model.albef.text_encoder.encoder.layer)):
            in_f, out_f = self.albef_model.albef.text_encoder.encoder.layer[i].attention.self.query.weight.shape[:2]
            self.albef_model.albef.text_encoder.encoder.layer[i].attention.self.query = lora.Linear(in_f, out_f, r=16)
            self.albef_model.albef.text_encoder.encoder.layer[i].attention.self.value = lora.Linear(in_f, out_f, r=16)

        for i in range(len(self.albef_model.albef.text_decoder.bert.encoder.layer)):
            in_f, out_f = self.albef_model.albef.text_decoder.bert.encoder.layer[i].attention.self.query.weight.shape[:2]
            self.albef_model.albef.text_decoder.bert.encoder.layer[i].attention.self.query = lora.Linear(in_f, out_f, r=16)
            self.albef_model.albef.text_decoder.bert.encoder.layer[i].attention.self.value = lora.Linear(in_f, out_f, r=16)

        from src.modeling.adaptered_output import Attention_lorad
        for i in range(len(self.albef_model.albef.visual_encoder.blocks)):
            self.albef_model.albef.visual_encoder.blocks[i].attn = Attention_lorad(
                self.albef_model.albef.visual_encoder.blocks[i].attn,
                768
            )

    def set_active_adapter(self, name):
        for i in range(len(self.albef_model.albef.text_encoder.encoder.layer)):
            self.albef_model.albef.text_encoder.encoder.layer[i].output.adapter.set_active_adapter(name)

        for i in range(len(self.albef_model.albef.text_decoder.bert.encoder.layer)):
            self.albef_model.albef.text_decoder.bert.encoder.layer[i].output.adapter.set_active_adapter(name)

        for i in range(len(self.albef_model.albef.visual_encoder.blocks)):
            self.albef_model.albef.visual_encoder.blocks[i].adapter.set_active_adapter(name)

    def deactivate_gating(self):
        for i in range(len(self.albef_model.albef.text_encoder.encoder.layer)):
            self.albef_model.albef.text_encoder.encoder.layer[i].output.adapter.deactivate_gating()

        for i in range(len(self.albef_model.albef.text_decoder.bert.encoder.layer)):
            self.albef_model.albef.text_decoder.bert.encoder.layer[i].output.adapter.deactivate_gating()

        for i in range(len(self.albef_model.albef.visual_encoder.blocks)):
            self.albef_model.albef.visual_encoder.blocks[i].adapter.deactivate_gating()

    def activate_gating(self):
        for i in range(len(self.albef_model.albef.text_encoder.encoder.layer)):
            self.albef_model.albef.text_encoder.encoder.layer[i].output.adapter.activate_gating()

        for i in range(len(self.albef_model.albef.text_decoder.bert.encoder.layer)):
            self.albef_model.albef.text_decoder.bert.encoder.layer[i].output.adapter.activate_gating()

        for i in range(len(self.albef_model.albef.visual_encoder.blocks)):
            self.albef_model.albef.visual_encoder.blocks[i].adapter.activate_gating()

    def forward(self, task_key: str, batch: Dict):
        """
        Does forward pass of image and text inputs through model,

        Args:
        task_key - string which indicates which task to do forward pass for

        Returns:
        https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling
        """

        # task_config = self.task_configs[task_key]
        output = self.albef_model(batch)
        return output


def load_albef(logger, model_config, checkpoint_name: str, device: torch.device, pretrained_albef_name: str, ) -> ALBEFWrapper:
    """
    Method to load ALBEFWrapper, around specified pre-trained albef

    args:
    checkpoint_name: name of ALBEF checkpoint to load encoder from
    device: torch.device
    pretrained_albef_name: pretrained albef name for processor/config

    returns:
    albef_model: ALBEFWrapper initialized with checkpoint
    """
    logger.info("-" * 100)
    logger.info("Loading ALBEF model: {}".format(checkpoint_name))

    BERT_LOCAL_PATH = './models/bert-base-uncased'
    logger.info("Loading tokenizer from: {}".format(BERT_LOCAL_PATH))
    tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH, local_files_only=True)

    if checkpoint_name == pretrained_albef_name:  # load pretrained albef model
        logger.info("Loading pretrained ALBEF model: {}".format(pretrained_albef_name))
        model = ALBEF(config=model_config, text_encoder=model_config["text_encoder"], text_decoder=model_config["text_decoder"], tokenizer=tokenizer, )

        checkpoint = torch.load(pretrained_albef_name, map_location="cpu")
        state_dict = checkpoint["model"]
        # todo: if load pre-fine-tuned albef, comment out the following two lines
        pos_embed_reshaped = interpolate_pos_embed(state_dict["visual_encoder.pos_embed"], model.visual_encoder)
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped

        # todo: if load pre-fine-tuned albef, comment out the following if
        if model_config["distill"]:
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m)
            state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if "bert" in key:
                encoder_key = key.replace("bert.", "")
                state_dict[encoder_key] = state_dict[key]
                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
            if "text_encoder" in key:
                if "layer" in key:
                    encoder_keys = key.split(".")
                    layer_num = int(encoder_keys[4])
                    if layer_num < 6:
                        del state_dict[key]
                        continue
                    else:
                        decoder_layer_num = layer_num - 6
                        encoder_keys[4] = str(decoder_layer_num)
                        encoder_key = ".".join(encoder_keys)
                else:
                    encoder_key = key
                decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                state_dict[decoder_key] = state_dict[key]

                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)

        albef_model = ALBEFWrapper(model, device)

    else:
        raise ValueError("Checkpoint name {} not supported".format(checkpoint_name))

    logger.info("Successfully loaded pretrained ALBEF")
    return albef_model


def create_albef_continual_learner_model(logger, model_name_or_path: str, ordered_cl_tasks: List[str], model_config: Dict, task_configs: Dict, device: torch.device, ):
    """
    Creates an instance of ALBEFContinualLearner, with the encoder initialized from model_name_or_path

    Args:
    model_name_or_path: Name/path of model to load encoder checkpoint from
    ordered_cl_tasks: List of task_keys to do continual learning on
    model_config: Dictionary containing ALBEF model configuration
    task_configs: Dictionary containing task-specific configurations for the CL tasks
    device: cpu/cuda

    Returns:
    cl_model: instance of ALBEFContinualLearner
    """

    albef_model = load_albef(logger, model_config=model_config, checkpoint_name=model_name_or_path, device=device, pretrained_albef_name=model_name_or_path, )

    cl_model = ALBEFContinualLearner(ordered_cl_tasks=ordered_cl_tasks, albef_model=albef_model, task_configs=task_configs, )
    logger.info("Successfully created and initialized ALBEF Continual Leaner model")

    return cl_model


def convert_batch_to_albef_input_dict(batch: Dict):
    """
    Convert inputs from batch_col
    late into format consumable by the ViltProcessor
    """
    return {
        "images": batch[0],
        "questions": batch[1],
        "answers": batch[2],
        "weights": batch[3],
        "n": batch[4],
        "alpha": batch[5], }

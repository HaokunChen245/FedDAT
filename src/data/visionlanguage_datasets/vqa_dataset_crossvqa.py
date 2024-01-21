import copy
import sys
import os
import time
import json
import logging
import random
import re
import glob
import base64
from tqdm import tqdm
from collections import defaultdict, Counter
from torchvision import transforms
import pickle as pkl
import pdb
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image
from src.utils.image_utils import resize_image
from src.utils.vqa_utils import get_score, target_tensor

from src.data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from src.data.image_collation import image_collate


class VQADataset(Dataset):
    def __init__(
        self,
        logger,
        data_dir: str,
        images_dataset: MSCOCOImagesDataset,
        split: str,
        task_key: str,
        encoder_type,
        transform=None,
        **kwargs
    ):

        """
        Initiates the VQADataset - loads all the questions (and converts to input IDs using the tokenizer, if provided)
        and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single QA pair, with a corresponding image

        Args:
        data_dir : path containing VQA questions and annotations. Also contains mapping from each answer in set of possible answers to a numerical label
        images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image for each question
        split: either train/val split

        Returns:
        Loads all annotations into self.data, where each item is a single VQA pair
        """

        self.images_dataset = images_dataset
        if transform:
            self.images_dataset.pil_transform = transform
            self.images_dataset.use_albef = True
        self.data_dir = data_dir
        self.encoder_type = encoder_type
        if split=='test':
            split = 'test_small'
        self.split = split
        self.task_key = task_key

        file_root =  "./data/"

        self.tokenizer = kwargs["tokenizer"] if "tokenizer" in kwargs else None
        self.label2idxs = {}
        if "abstract" in task_key:
            self.questions_file = os.path.join(
                data_dir, "abstract_{}.json".format(split)
            )
            self.annotations_file = os.path.join(
                data_dir, "abstract_v002_val2015_annotations.json".format(split)
            )
            self.ans2label_file = os.path.join(file_root, "abstract/ans2label.pkl")
        elif "toronto" in task_key:
            self.annotations_file = os.path.join(
                data_dir, "toronto_{}.json".format(split)
            )
            self.questions_file = os.path.join(
                data_dir, "toronto_{}.json".format(split)
            )
            self.ans2label_file = os.path.join(file_root, "toronto/ans2label.pkl".format(split))
        elif "art" in task_key:
            self.annotations_file = os.path.join(file_root, "art/art_{}.json".format(split))
            self.questions_file = os.path.join(file_root, "art/art_{}.json".format(split))
            self.ans2label_file = os.path.join(file_root, "art/ans2label_small.pkl".format(split))
        elif "gqa" in task_key:
            self.ans2label_file = file_root + "GQA/ans2label_fed.pkl"
        elif "vizwiz" in task_key:
            self.ans2label_file = file_root + "vizwiz/ans2label_fed.pkl"
        elif "clove_scene" in task_key:
            scene_key = task_key.replace("clove_", "")
            root = "/CLOVE/json/scene"
            for fname in os.listdir(root):
                if scene_key in fname and 'ans2label' in fname:
                    break
            self.ans2label_file = os.path.join(root, fname)
        elif "clove_function" in task_key:
            k = task_key.replace("clove_function_", "")
            function_key = {"a": "attribute",
                 "b": "knowledge",
                 "c": "logical",
                 "d": "object",
                 "e": "relation",
            }[k]

            root = file_root + "/CLOVE/json/function"
            for fname in os.listdir(root):
                if function_key in fname and 'ans2label' in fname:
                    break
            self.ans2label_file = os.path.join(root, fname)

        # Load mapping from answers to labels
        self.ans2label = pkl.load(open(self.ans2label_file, "rb"))
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        self.num_labels = 100 # len(self.label2ans)

        self.cached_data_file = os.path.join(
            self.data_dir, "cached_vqa_data", "vqa_{}.pkl".format(split)
        )
        if task_key in ["gqa", "vizwiz"]:
            self.cached_data_file = os.path.join(
                self.data_dir, "{}_fed.pkl".format(split.split('_')[0])
            )
        elif "clove" in task_key:
            if "test" in split:
                self.cached_data_file = self.ans2label_file.replace("ans2label", "val")
            else:
                self.cached_data_file = self.ans2label_file.replace("ans2label", split.split('_')[0])

        if os.path.isfile(self.cached_data_file):
            # Load cached data
            # self.data = pkl.load(open(self.cached_data_file, "rb"))
            # if not os.path.exists(self.cached_data_file):
            if task_key not in ["gqa", "vizwiz"] and "clove" not in task_key:
                p = self.cached_data_file.replace('.', '_fed.')
            else:
                p = self.cached_data_file
            self.data = pkl.load(open(p, "rb"))
            for d in self.data:
                if "question_input_ids" not in d.keys():
                    d["question_input_ids"] = []
            random.shuffle(self.data)
            # ct = 0
            # temp = []
            # for d in self.data:
            #     f = True
            #     for l in d["labels"]:
            #         if l>=100:
            #             f = False
            #             break
            #     if f and len(d["labels"])>0:
            #         temp.append(d)
            # self.data = []
            # for d in temp:
            #     if 'train' in self.split:
            #         if random.random() <= 2500.0/len(temp):
            #             self.data.append(d)
            #     else:
            #         if random.random() <= 500.0/len(temp):
            #             self.data.append(d)
            # pkl.dump(self.data, open(p, "wb"))

        else:
            # Create map from question id to question
            # vqav2 & abstractqq
            # questions = json.load(open(self.questions_file))['questions']
            questions = json.load(open(self.questions_file))
            qid2qdata = {x["question_id"]: x for x in questions}

            # Create data for each annotation
            # vqav2 & abstract
            # annotations = json.load(open(self.annotations_file))['annotations']
            annotations = json.load(open(self.annotations_file))
            self.data = []
            # annotations_dict = {x['question_id']: x for x in annotations}
            # for ques in questions:
            #   qid = ques['question_id']
            #   image_id = int(ques['image'].split('/')[-1].split('.')[0].split('_')[-1])
            #   anno = annotations_dict[qid]
            #   assert image_id == anno['image_id']
            for anno in annotations:
                qid = anno["question_id"]
                # vqav2 & abstract
                # image_id = int(anno['image'].split('/')[-1].split('.')[0].split('_')[-1])
                # pvqa
                image_id = anno["image"].split("/")[-1].split(".")[0]
                # image_id = anno['image'].strip('.jpg').split('/')[-1]
                # image_id = int(anno['image'].strip('.jpg').split('-')[0])

                # Retrieve the question for this annotation
                qdata = qid2qdata[qid]
                # assert qdata['image_id'] == image_id
                # qdata_img_id = int(qdata['image'].split('/')[-1].split('.')[0].split('_')[-1])
                # pvqa
                qdata_img_id = qdata["image"].split("/")[-1].split(".")[0]
                # qdata_img_id = qdata['image'].strip('.jpg').split('/')[-1]
                # qdata_img_id = int(qdata['image'].strip('.jpg').split('-')[0])
                assert qdata_img_id == image_id
                question = qdata["question"]
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(question)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = []
                    input_ids = []

                # Map from each crowdsourced answer to occurrences in annotation
                # answers = [a['answer'] for a in anno['answers']]
                answers = anno["answer"]
                answer_count = defaultdict(int)
                for ans in answers:
                    answer_count[ans] += 1

                # Get label and score (0.3/0.6/1) corresponding to each crowdsourced answer
                labels = []
                scores = []
                answers = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    if task_key in ["toronto", "pvqa", "med", "art", "gqa"] or "clova" in task_key:
                        score = 1 / answer_count[answer]
                    else:
                        score = get_score(answer_count[answer])
                    scores.append(score)
                    answers.append(answer)
                correct_answer = answers[0]

                # Store pre-processed example
                example = {
                    "question_id": qid,
                    "image_id": image_id,
                    "question": question,
                    "question_input_ids": input_ids,
                    "correct_answer": correct_answer,
                    "labels": labels,
                    "answers": answers,
                    "scores": scores,
                }
            # if not os.path.isdir(self.cached_data_file):
            #     os.makedirs(self.cached_data_file)
            pkl.dump(self.data, open(self.cached_data_file, "wb"))

        self.n_examples = len(self.data)
        # for data in self.data:
        #    data['correct_answer'] = data['correct_answer'][0]
        # pkl.dump(self.data, open(self.cached_data_file, 'wb'))
        if task_key == "abstract":
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "Loaded VQA abstract {} dataset, len:{}".format(
                        self.split, len(self.data)
                    )
                )
        elif task_key == "toronto":
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "Loaded toronto {} dataset, len:{}".format(
                        self.split, len(self.data)
                    )
                )
        elif task_key == "med":
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "Loaded Med2019 {} dataset, len:{}".format(
                        self.split, len(self.data)
                    )
                )
        elif task_key == "pvqa":
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "Loaded PathVQA {} dataset, len:{}".format(
                        self.split, len(self.data)
                    )
                )
        elif task_key == "art":
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "Loaded AQUA {} dataset, len:{}".format(
                        self.split, len(self.data)
                    )
                )
        else:
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "Loaded VQAv2 {} dataset, len:{}".format(
                        self.split, len(self.data)
                    )
                )

        if self.encoder_type == 'albef':
            self.answer_list = list(self.ans2label.keys())[:100]
            self.transform = transform
            self.eos = "[SEP]"
            self.max_ques_words = 30

            if not 'train' in self.split:
                self.max_ques_words = 50  # do not limit question length during test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do VQA

        """
        if self.encoder_type == "vilt":
            example  = self.data[index]
            question_id = example["question_id"]

            # Tokenize the input question
            question = example["question"]
            input_ids = example["question_input_ids"]

            # Get the image tensor from ImageDataset
            image_id = example["image_id"]
            image = self.images_dataset.get_image_data(image_id)

            labels = example["labels"]
            scores = example["scores"]
            target_scores = target_tensor(self.num_labels, labels, scores)

            return {
                "question": question,
                "input_ids": input_ids,
                "image": image,
                "labels": labels,
                "target_scores": target_scores,
                "question_id": question_id,
            }

        else:

            # for albef
            ann = self.data[index]
            image = self.images_dataset.get_image_data(ann["image_id"])
            if not 'train' in self.split:
                question = pre_question(ann["question"], self.max_ques_words)
                if self.task_key == 'abstract' or self.task_key == 'art':
                    while len(ann["labels"])<10:
                        ann["labels"].append(-1)
                gt = ann["labels"]
                if not isinstance(ann["labels"], list):
                    gt = [gt]
                gt = torch.Tensor(gt).long()
                return image, question, gt

            else:
                question = pre_question(ann["question"], self.max_ques_words)
                answer_weight = {}
                for answer in ann["answers"]:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann["answers"])
                    else:
                        answer_weight[answer] = 1 / len(ann["answers"])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())
                answers = [answer + self.eos for answer in answers]
                return image, question, answers, weights

def vqa_batch_collate(batch: List[Dict], visual_input_type: str):
    """
    Collates each model input for all batch items into a single model input (e.g. converts a list of input_ids into a matrix of size (batch_size, max_len))

    Args:
    batch - list of batch items, each item being a dictionary returned by Dataset's __getitem__ method
    visual_input_type: string which specifies the type of visual input

    Returns:
    Dictionary containing batched inputs and outputs
    """

    pad_token = 0  # tokenizer.pad_token_id

    # Pad the text inputs
    questions = [x["question"] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    max_len = max([len(x) for x in input_ids])
    input_ids_padded = []
    attn_masks = []
    for i in range(len(input_ids)):
        ids_padded = input_ids[i] + [pad_token] * (max_len - len(input_ids[i]))
        attn_mask = [1] * len(input_ids[i]) + [0] * (max_len - len(input_ids[i]))

        input_ids_padded.append(ids_padded)
        attn_masks.append(attn_mask)
    input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long)

    # Stack the target tensors
    batch_labels = [x["labels"] for x in batch]
    batch_scores = [x["target_scores"] for x in batch]
    batch_scores = torch.stack(batch_scores, dim=0)

    # Depending on the visual_input_type variable, process the images accordingly
    images = [x["image"] for x in batch]
    images = image_collate(images, visual_input_type)

    return {
        "raw_texts": questions,
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "images": images,
        "target_scores": batch_scores,
        "labels": batch_labels,
    }

def pre_question(question, max_ques_words):
    question = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            question.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question

def vqa_collate_fn_eval(batch):
    # this function is used for ALBEF
    image_list, question_list, answer_list = [], [], []
    for image, question, answer in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list.append(answer)
    return [
        torch.stack(image_list, dim=0),
        question_list,
        torch.stack(answer_list, dim=0),
    ]

def vqa_collate_fn(batch):
    # this function is used for ALBEF
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer #[1,2,3] [2,3,4] [1,2,3,2,3,4]
        n.append(len(answer))
    return [
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    ]


def build_vqa_vilt_dataloader(
    logger,
    args,
    data_dir: str,
    images_dataset: MSCOCOImagesDataset,
    split: str,
    task_key: str,
    visual_input_type: str,
    client_id=-1,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Creates the VQA Dataloader, which gives batches of VQA inputs and outputs

    Args:
    data_dir : path containing VQA questions and annotations.
    images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image for each question
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    batch_size = args.batch_size
    shuffle = True if split == "train" or "train_small" else False

    dataset = VQADataset(logger, data_dir, images_dataset, split, task_key, "vilt", client_id=client_id, **kwargs)
    if torch.distributed.get_rank() == 0:
        logger.info(
            "Dataset for ViLT domain:{}, split:{}, len:{}, bs:{}".format(
                task_key, split, len(dataset), batch_size
            )
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: vqa_batch_collate(x, visual_input_type),
    )
    return dataloader


def build_vqa_albef_dataloader(
    logger, args, data_dir, images_dataset, vqa_config, split: str, task_key: str, client_id=-1, **kwargs
) -> torch.utils.data.DataLoader:
    """
    Creates the VQA Dataloader, which gives batches of VQA inputs and outputs

    Args:
    split: either train/val split
    visual_input_type: format of visual input to model

    Returns:
    DataLoader object
    """

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(
            #     384, scale=(0.5, 1.0), interpolation=Image.BICUBIC
            # ),
            # transforms.RandomHorizontalFlip(),
            # RandomAugment(
            #     2,
            #     7,
            #     isPIL=True,
            #     augs=[
            #         "Identity",
            #         "AutoContrast",
            #         "Equalize",
            #         "Brightness",
            #         "Sharpness",
            #         "ShearX",
            #         "ShearY",
            #         "TranslateX",
            #         "TranslateY",
            #         "Rotate",
            #     ],
            # ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            # transforms.Resize(size=384, max_size=640),
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if "train" in split:
        transform = train_transform
        shuffle = True
        drop_last = True
        collate_fn = vqa_collate_fn
        batch_size = args.batch_size
    else:
        transform = test_transform
        shuffle = False
        drop_last = False
        collate_fn = vqa_collate_fn_eval
        batch_size = args.val_batch_size

    dataset = VQADataset(logger, data_dir, images_dataset, split, task_key, "albef", transform, client_id=client_id)

    if torch.distributed.get_rank() == 0:
        logger.info(
            "Created ALBEF VQA {} {} dataloader with len of {}, batch size of {}".format(
                task_key, split, len(dataset), batch_size
            )
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    return dataloader


if __name__ == "__main__":
    data_dir = "/data/datasets/MCL/vqav2/"

    class Args:
        def __init__(self):
            self.batch_size = 4
            self.shuffle = True
            self.num_workers = 2
            self.visual_input_type = "pil-image"

    args = Args()

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    images_dataset = MSCOCOImagesDataset(
        "/data/datasets/MCL/ms-coco/", args.visual_input_type
    )
    vqa_dataloader = build_vqa_vilt_dataloader(
        args,
        data_dir,
        images_dataset,
        "val",
        args.visual_input_type,
        tokenizer=tokenizer,
    )

    for batch in vqa_dataloader:
        pdb.set_trace()

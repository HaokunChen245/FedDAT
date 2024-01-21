from src.train.visionlanguage_tasks.train_vqa import VQATrainer
from src.train.visionlanguage_tasks.train_vqa_crossvqa import VQATrainerCross
from src.train.visionlanguage_tasks.train_nlvr2 import NLVR2Trainer
from src.train.visionlanguage_tasks.train_snli_ve import SNLIVETrainer
from src.train.visionlanguage_tasks.train_vcr import VCRTrainer
import copy

SUPPORTED_VL_TASKS = [
    "vqa",
    "abstract",
    "toronto",
    "vizwiz",
    "pvqa",
    "med",
    "art",
] + ['vqa', 'nlvr2', 'snli-ve', 'vcr']

SUPPORTED_ORDER = ["Order1", "Order2", "Order3", "Order4", "Order5", "Debug_order"]

data_root = "./data"

mscoco_config = {
    "data_dir": data_root + "/mscoco",
}
abstract_image_config = {
    "data_dir": [
        data_root + "/vqa_abstract/train2015",
        data_root + "/vqa_abstract/val2015",
    ],
}
toronto_image_config = {
    "data_dir": [
        data_root + "/mscoco/train2014",
        data_root + "/mscoco/val2014",
    ]
}
art_image_config = {"data_dir": [data_root + "/AQUA/SemArt/Images"]}

clove_function_a_config = {
    "task_name": "clove_function_a",
    "data_dir": data_root + "/CLOVE/json/function",
    "images_source": "vgd",
    "splits": ["train", "val_small"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,
}

clove_function_b_config = copy.deepcopy(clove_function_a_config)
clove_function_b_config["task_name"] = "clove_function_b"
clove_function_c_config = copy.deepcopy(clove_function_a_config)
clove_function_c_config["task_name"] = "clove_function_c"
clove_function_d_config = copy.deepcopy(clove_function_a_config)
clove_function_d_config["task_name"] = "clove_function_d"
clove_function_e_config = copy.deepcopy(clove_function_a_config)
clove_function_e_config["task_name"] = "clove_function_e"

clove_scene_a_config = {
    "task_name": "clove_scene_a",
    "data_dir": data_root + "/CLOVE/json/scene",
    "images_source": "vgd",
    "splits": ["train", "val_small"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,
}

clove_scene_b_config = copy.deepcopy(clove_scene_a_config)
clove_scene_b_config["task_name"] = "clove_scene_b"
clove_scene_c_config = copy.deepcopy(clove_scene_a_config)
clove_scene_c_config["task_name"] = "clove_scene_c"
clove_scene_d_config = copy.deepcopy(clove_scene_a_config)
clove_scene_d_config["task_name"] = "clove_scene_d"
clove_scene_e_config = copy.deepcopy(clove_scene_a_config)
clove_scene_e_config["task_name"] = "clove_scene_e"
clove_scene_f_config = copy.deepcopy(clove_scene_a_config)
clove_scene_f_config["task_name"] = "clove_scene_f"

vizwiz_config = {
    "task_name": "vizwiz",
    "data_dir": data_root + "/vizwiz",
    "images_source": "vizwiz",
    "splits": ["train", "val_small"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,
}

gqa_config = {
    "task_name": "gqa",
    "data_dir": data_root + "/GQA",
    "images_source": "vg",
    "splits": ["train", "val_small"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,
}

abstract_config = {
    "task_name": "abstract",
    "data_dir": data_root + "/vqa_abstract",
    "images_source": "abstract_image",
    "splits": ["train", "val_small"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,
}


toronto_config = {
    "task_name": "toronto",
    "data_dir": data_root + "/torontoCOCO",
    "images_source": "toronto_image",
    "splits": ["train", "val"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,
}

art_config = {
    "task_name": "art",
    "data_dir": data_root + "/albef/art",
    "images_source": "art_image",
    "splits": ["train", "val"],
    "num_labels": 100,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainerCross,
    "random_baseline_score": 0.0,

}

mscoco_config = {
        'data_dir': 'ms-coco/',
}

flickr_config = {
    'data_dir': 'flickr30k/',
}

vqa_config = {
        'task_name': 'VQAv2',
        'data_dir': 'vqav2/',
        'images_source': 'ms-coco',
        'splits': ['train', 'val'],
        'num_labels': 3129,
        'num_images': 1,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'task_trainer': VQATrainerCross,
        'random_baseline_score': 0.0,
}

nlvr_config = {
        'task_name': 'NLVRv2',
        'data_dir': 'nlvr2/',
        'splits': ['train', 'val'],
        'num_labels': 2,
        'num_images': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'task_trainer': NLVR2Trainer,
        'random_baseline_score': 50.0,
}

snli_ve_config = {
        'task_name': 'SNLI-VE',
        'data_dir': 'snli-ve/',
        'images_source': 'flickr30k',
        'splits': ['train', 'dev', 'test'],
        'num_labels': 3,
        'num_images': 1,
        'model_type': 'classification',
        'num_epochs': 5,
        'lr': 5e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'task_trainer': SNLIVETrainer,
        'random_baseline_score': 33.33,
}

vcr_config = {
        'task_name': 'VCR',
        'data_dir': 'vcr/',
        'splits': ['train', 'dev', 'test'],
        'num_labels': 4,
        'num_images': 1,
        'model_type': 'multi-choice',
        'task_type': 'answer',
        'num_choices': 4,
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'task_trainer': VCRTrainer,
        'random_baseline_score': 25.0,
}

task_configs = {
    "ms-coco": mscoco_config,
    "flickr30k": flickr_config,
    "abstract": abstract_config,
    "clove_scene_a": clove_scene_a_config,
    "clove_scene_b": clove_scene_b_config,
    "clove_scene_c": clove_scene_c_config,
    "clove_scene_d": clove_scene_d_config,
    "clove_scene_e": clove_scene_e_config,
    "clove_scene_f": clove_scene_f_config,
    "clove_function_a": clove_function_a_config,
    "clove_function_b": clove_function_b_config,
    "clove_function_c": clove_function_c_config,
    "clove_function_d": clove_function_d_config,
    "clove_function_e": clove_function_e_config,
    "toronto": toronto_config,
    "vizwiz": vizwiz_config,
    "gqa": gqa_config,
    "art": art_config,
    'vqa': vqa_config,
    'nlvr2': nlvr_config,
    'snli-ve': snli_ve_config,
    'vcr': vcr_config,
    "abstract_image": abstract_image_config,
    "toronto_image": toronto_image_config,
    "art_image": art_image_config,
}

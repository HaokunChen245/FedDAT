from src.train.visionlanguage_tasks.train_vqa import VQATrainer
from src.train.visionlanguage_tasks.task_trainer import TaskTrainer

SUPPORTED_VL_TASKS = [
    "vqa",
    "abstract",
    "toronto",
    "pvqa",
    "med",
    "art",
    "abstract_albef",
    "toronto_albef",
    "pvqa_albef",
    "med_albef",
    "art_albef",
]
SUPPORTED_ORDER = ["Order1", "Order2", "Order3", "Order4", "Order5", "Debug_order"]

mscoco_config = {
    "data_dir": "/nfs/data3/zhangya/albef/mscoco/",
}
abstract_image_config = {
    "data_dir": [
        "/nfs/data3/zhangya/vqa_abstract/train2015",
        "/nfs/data3/zhangya/vqa_abstract/val2015",
    ],
}
toronto_image_config = {
    "data_dir": [
        "/nfs/data3/zhangya/albef/mscoco/train2014",
        "/nfs/data3/zhangya/albef/mscoco/val2014",
    ]
}

pvqa_image_config = {
    "data_dir": [
        "/nfs/data3/zhangya/PathVQA_data/PathVQA/split/images/train",
        "/nfs/data3/zhangya/albef/pvqa/split/images/val",
        "/nfs/data3/zhangya/albef/pvqa/split/images/test",
    ]
}
med_image_config = {"data_dir": ["/nfs/data3/yyang/med_images/all_images"]}
art_image_config = {"data_dir": ["/nfs/data3/yyang/AQUA/SemArt/Images"]}

flickr_config = {
    "data_dir": "flickr30k/",
}

vqa_config = {
    "task_name": "VQAv2",
    "data_dir": "vqav2/",
    "images_source": "ms-coco",
    "splits": ["train", "val"],
    "num_labels": 3129,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 50,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,
}

abstract_config = {
    "task_name": "abstract",
    "data_dir": "/nfs/data3/zhangya/vqa_abstract",
    "images_source": "abstract_image",
    "splits": ["train", "val_small"],
    "num_labels": 500,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,
}

toronto_config = {
    "task_name": "toronto",
    "data_dir": "/nfs/data3/zhangya/torontoCOCO",
    "images_source": "toronto_image",
    "splits": ["train", "val"],
    "num_labels": 430,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,
}
pvqa_config = {
    "task_name": "pvqa",
    "data_dir": "/nfs/data3/zhangya/albef/pvqa",
    "images_source": "pvqa_image",
    "splits": ["train", "val"],
    "num_labels": 2540,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,
}

med_config = {
    "task_name": "med",
    "data_dir": "/nfs/data3/zhangya/VQA-Med-2019",
    "images_source": "med_image",
    "splits": ["train", "val"],
    "num_labels": 1701,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,
}

art_config = {
    "task_name": "art",
    "data_dir": "/nfs/data3/zhangya/albef/art",
    "images_source": "art_image",
    "splits": ["train", "val"],
    "num_labels": 326,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,

}

abstract_albef_config = {
    "task_name": "abstract",
    "train": "/nfs/data3/zhangya/vqa_abstract/vqa_abstract_train.json",
    "train_small": "/nfs/data3/zhangya/vqa_abstract/abstract_train_small.json",
    "val": "/nfs/data3/zhangya/vqa_abstract/abstract_val.json",
    "test": "/nfs/data3/zhangya/vqa_abstract/abstract_test_small.json",
    "images": "/nfs/data3/zhangya/vqa_abstract",
    "answer_list": "/nfs/data3/zhangya/vqa_abstract/answer_list.json",
    "splits": ["train", "val"],
    # 'num_labels': 500,
    # 'num_images': 1,
    # 'model_type': 'classification',
    "num_epochs": 20,  # Yao: original 10
    # 'lr': 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    # 'warmup_ratio': 0.1,
    "task_trainer": VQATrainer,
    # 'random_baseline_score': 0.0,
    # 'low_shot_config': {'task_trainer': LowShotVQATrainer,
    #                     'type': 'percentage',
    #                     'percentage': 0.05,
    #                     'eval_epochs': [6, 8, 10, 15, 20, 25, 30]}
}

toronto_albef_config = {
    "task_name": "toronto",
    "train": "/nfs/data3/zhangya/torontoCOCO/toronto_train.json",
    "train_small": "/nfs/data3/zhangya/torontoCOCO/toronto_train_small.json",
    "val": "/nfs/data3/zhangya/torontoCOCO/toronto_val.json",
    "test": "/nfs/data3/zhangya/torontoCOCO/toronto_test_small.json",
    "images": "/nfs/data3/zhangya/albef/mscoco",
    "answer_list": "/nfs/data3/zhangya/torontoCOCO/answer_list.json",
    "splits": ["train", "val"],
    # 'num_labels': 430,
    # 'num_images': 1,
    # 'model_type': 'classification',
    "num_epochs": 20,  # Yao: original 10
    # 'lr': 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    # 'warmup_ratio': 0.1,
    "task_trainer": VQATrainer,
    # 'random_baseline_score': 0.0,
    # 'low_shot_config': {'task_trainer': LowShotVQATrainer,
    #                     'type': 'percentage',
    #                     'percentage': 0.05,
    #                     'eval_epochs': [6, 8, 10, 15, 20, 25, 30]}
}
pvqa_albef_config = {
    "task_name": "pvqa",
    "train": "/nfs/data3/zhangya/albef/pvqa/pvqa_train.json",
    "train_small": "/nfs/data3/zhangya/albef/pvqa/pvqa_train_small.json",
    "val": "/nfs/data3/zhangya/albef/pvqa/pvqa_val.json",
    "test": "/nfs/data3/zhangya/albef/pvqa/pvqa_test_small.json",
    "answer_list": "/nfs/data3/zhangya/albef/pvqa/answer_list_small.json",
    "images": "/nfs/data3/zhangya/PathVQA_data/PathVQA/split/images",
    "splits": ["train", "val"],
    "num_labels": 2540,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,
}

med_albef_config = {
    "task_name": "med",
    "train": "/nfs/data3/zhangya/VQA-Med-2019/med2019_train.json",
    "train_small": "/nfs/data3/zhangya/VQA-Med-2019/med2019_train.json",
    "val": "/nfs/data3/zhangya/VQA-Med-2019/med2019_val.json",
    "test": "/nfs/data3/zhangya/VQA-Med-2019/med2019_test.json",
    "answer_list": "/nfs/data3/zhangya/VQA-Med-2019/answer_list_trainval.json",
    "images": "/nfs/data3/zhangya/VQA-Med-2019",
    "splits": ["train", "val"],
    "num_labels": 1701,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,  # Yao: original 10
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": VQATrainer,
    "random_baseline_score": 0.0,

}

art_albef_config = {
    "task_name": "art",
    "train": "/nfs/data3/zhangya/albef/art/art_train.json",
    "train_small": "/nfs/data3/zhangya/albef/art/art_train_small.json",
    "val": "/nfs/data3/zhangya/albef/art/art_val.json",
    "test": "/nfs/data3/zhangya/albef/art/art_test_small.json",
    "images": "/nfs/data3/yyang/AQUA/SemArt/Images",
    "answer_list": "/nfs/data3/zhangya/albef/art/answer_list_small.json",
    "splits": ["train", "val"],
    # 'num_labels': 326,
    # 'num_images': 1,
    # 'model_type': 'classification',
    "num_epochs": 20,  # Yao: original 10
    # 'lr': 1e-4,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    # 'warmup_ratio': 0.1,
    "task_trainer": VQATrainer,
    # 'random_baseline_score': 0.0,
    # 'low_shot_config': {'task_trainer': LowShotVQATrainer,
    #                     'type': 'percentage',
    #                     'percentage': 0.05,
    #                     'eval_epochs': [6, 8, 10, 15, 20, 25, 30]}

}

task_configs = {
    "ms-coco": mscoco_config,
    "flickr30k": flickr_config,
    "vqa": vqa_config,
    "abstract": abstract_config,
    "toronto": toronto_config,
    "pvqa": pvqa_config,
    "med": med_config,
    "art": art_config,
    "abstract_albef": abstract_albef_config,
    "toronto_albef": toronto_albef_config,
    "pvqa_albef": pvqa_albef_config,
    "med_albef": med_albef_config,
    "art_albef": art_albef_config,
    "abstract_image": abstract_image_config,
    "toronto_image": toronto_image_config,
    "pvqa_image": pvqa_image_config,
    "med_image": med_image_config,
    "art_image": art_image_config,
}

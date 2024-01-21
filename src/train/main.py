import argparse
import json
import logging
import os
import sys
import copy
import types
import torch.nn as nn

sys.path.insert(0, ".")

import torch

from transformers.adapters import AdapterConfig

import sys

sys.path.append("/home/stud/yyang/CARVEN")

from src.modeling import create_continual_learner_map
from src.modeling.continual_learner import ContinualLearner

from src.configs.model_configs import model_configs, ALLOWED_CL_ENCODERS
from src.configs.task_configs_fed import (
    task_configs,
    SUPPORTED_VL_TASKS,
    SUPPORTED_ORDER,
)
from src.configs.adapter_configs import ADAPTER_MAP

import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs

def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def get_average_net(server, c_models, nums, ordered_tasks, device):
    total_num_train = sum(nums)
    with torch.no_grad():
        for key in server.comm_state_dict_names:
            if 'clf' in key:
                continue
            if "num_batches_tracked" in key:
                server.state_dict()[key].data.copy_(c_models[0][key])
            else:
                # server_adapter will also be averaged
                temp = torch.zeros_like(server.state_dict()[key]).float().to(device)
                for net, num in zip(c_models, nums):
                    temp += net[key] * num / total_num_train
                server.state_dict()[key].data.copy_(temp)

    return server

def root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split(".")[0])
    # if the logger has been initialized, just return it
    # if logger.hasHandlers():
    #     return logger

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger

def prepare_model(args, logger):
    create_model_method = create_continual_learner_map[args.encoder_name]
    model_config = model_configs[args.encoder_name]

    if 'dat' in args.optimizer_mode:
        adapter_config = {}
        adapter_keys = []
        for i in range(3):
            adapter_keys += [f"adapter_{i}"]
        adapter_config['names'] = adapter_keys
        adapter_config['device'] = 'cuda'
        model_config['adapter_config'] = adapter_config

    elif 'adapter' in args.optimizer_mode:
        adapter_config = {}
        adapter_config['names'] = ['adapter']
        adapter_config['device'] = 'cuda'
        model_config['adapter_config'] = adapter_config

    model = create_model_method(logger=logger, model_name_or_path=args.pretrained_model_name,  # ContinualLearner
                                ordered_cl_tasks=args.ordered_cl_tasks,
                                model_config=model_configs[args.encoder_name],
                                task_configs=task_configs,
                                device='cuda')
    model.comm_state_dict_names = []

    if 'albef' in args.encoder_name:
        args.personal_params_names = ['.cls.']
    else:
        args.personal_params_names = ['task']

    if args.optimizer_mode=='full':
        for n, p in model.named_parameters():
            p.requires_grad = True
        for n in model.state_dict().keys():
            model.comm_state_dict_names.append(n)
    else:
        for n, p in model.named_parameters():
            p.requires_grad = False

    if args.optimizer_mode == "adapter":
        if 'vilt' in args.encoder_name:
            model.add_adapter()
        for n, p in model.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
        for n in model.state_dict().keys():
            if 'adapter' in n:
                model.comm_state_dict_names.append(n)

    elif "dat" in args.optimizer_mode:
        if 'vilt' in args.encoder_name:
            model.add_adapter()
        args.personal_params_names += ['adapter_0', 'adapter_2']
        args.shared_params_names = ['adapter_1']

        for n, p in model.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
        for n in model.state_dict().keys():
            for sn in args.shared_params_names:
                if sn in n:
                    model.comm_state_dict_names.append(n)

    elif args.optimizer_mode == "freeze_encoder":
        # Freeze encoder weights
        model.get_encoder().freeze_all_weights()

    elif args.optimizer_mode == "freeze_bottom_k_layers":
        # Freeze bottom K layers
        model.get_encoder().freeze_bottom_k_layers(k=args.layers_to_freeze)

    elif args.optimizer_mode=='none':
        pass

    elif args.optimizer_mode=='norm':
        for n, p in model.named_parameters():
            if 'norm' in n:
                p.requires_grad = True
        for n in model.state_dict().keys():
            if 'norm' in n:
                model.comm_state_dict_names.append(n)

    elif args.optimizer_mode=='lora':
        model.set_active_lora()
        for n in model.state_dict().keys():
            if 'lora' in n:
                model.comm_state_dict_names.append(n)

    elif args.optimizer_mode=='bias':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
        for n in model.state_dict().keys():
            if 'bias' in n:
                model.comm_state_dict_names.append(n)

    elif args.optimizer_mode=='prompt':
        if 'vilt' in args.encoder_name:
            from src.modeling.prompted_output import ViltEmbeddings_prompted_forward
            if 'bert' in args.encoder_name:
                m = model.viltbert_encoder.vilt.embeddings
            else:
                m = model.vilt_encoder.vilt.embeddings

            m.forward = types.MethodType(ViltEmbeddings_prompted_forward, m)
        else:
            from src.modeling.prompted_output import albef_prompted_forward, BERTEmbeddings_prompted_forward
            m = model.albef_model.albef
            m.forward = types.MethodType(albef_prompted_forward, m)
            # m_text = model.albef_model.albef.text_encoder
            # m_text.forward = types.MethodType(BERTEmbeddings_prompted_forward, m_text)

        prompt_len = 5
        m.prompt_tokens_vis = torch.arange(prompt_len).long()
        m.prompt_embedding_vis = nn.Sequential(
            nn.Embedding(prompt_len, 768),
            nn.Linear(768, 192),
            nn.Tanh(),
            nn.Linear(192, 768),
        )
        if 'vilt' in args.encoder_name:
            m.prompt_tokens_text = torch.arange(prompt_len).long()
            m.prompt_embedding_text = nn.Sequential(
                nn.Embedding(prompt_len, 768),
                nn.Linear(768, 192),
                nn.Tanh(),
                nn.Linear(192, 768),
            )
        else:
            pass
            # m_text.prompt_tokens_text = torch.arange(prompt_len).long()
            # m_text.prompt_embedding_text = nn.Sequential(
            #     nn.Embedding(prompt_len, 768),
            #     nn.Linear(768, 192),
            #     nn.Tanh(),
            #     nn.Linear(192, 768),
            # )

        for n, p in model.named_parameters():
            if 'prompt' in n:
                p.requires_grad = True
        for n in model.state_dict().keys():
            if 'prompt' in n:
                model.comm_state_dict_names.append(n)

    # training all task layers
    for n, p in model.named_parameters():
        if 'task' in n or '.cls.' in n:
            p.requires_grad = True

    # require grad params
    print('-----------------------------------')
    print('Require grad params:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    return model

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=ALLOWED_CL_ENCODERS,
                        help="The name of the base pretrained encoder.")
    parser.add_argument("--portion", default=1.0, type=float,
                        help="The name of optimization mode.")
    parser.add_argument("--optimizer_mode", default='none', type=str,
                        help="The name of optimization mode.")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True,
                        help="Name of pretrained model weights to load.")
    parser.add_argument("--climb_data_dir", type=str, required=True, default='/data/datasets/MCL/',
                        help="Directory where all the MCL data is stored")
    parser.add_argument("--debug", type=int, default=0,
                        help="If True, debug the code with minimum setting")
    parser.add_argument("--do_single", action='store_true',
                        help="If True, train the model on these tasks")
    parser.add_argument("--do_train", action='store_true',
                        help="If True, train the model on these tasks")
    parser.add_argument("--do_eval", action='store_true',
                        help="If True, evaluate the model on these tasks.")
    parser.add_argument("--do_test", action='store_true',
                        help="If True, evaluate pre-trained model on single task.")

    # Arguments specific to Adapters algorithm
    parser.add_argument("--adapter_config", choices=list(ADAPTER_MAP.keys()),
                        help="Type of Adapter architecture")
    parser.add_argument("--adapter_reduction_factor", type=int, default=0,
                        help="Downsampling ratio for adapter layers")
    # Arguments specific to frozen bottom-k layers algorithm
    parser.add_argument("--layers_to_freeze", type=int, default=0,
                        help="Number of layers to freeze (if freezing bottom-k layers)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Name of output directory, where all experiment results and checkpoints are saved.")
    parser.add_argument("--do_wandb_logging", action="store_true",
                        help="Log experiments in W&B.")
    parser.add_argument("--wandb_freq", type=int, default=100,
                        help="Log frequency in W&B.")
    parser.add_argument("--comm_rounds", type=int, default=20,
                        help="Number of communication rounds.")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="Number of communication rounds.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=15,
                        help="Maximum number of epochs to train.")
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help="Test Batch size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--ordered_cl_tasks", type=str)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to the checkpoint of singletask_ft, which is loaded for testing")
    parser.add_argument("--model_path", type=str, default=None,
                        help="path to the model for evaluation")

    args = parser.parse_args()
    args.visual_input_type = model_configs[args.encoder_name]["visual_input_type"]

    # --------------------- Set up experiment directories ---------------------
    setting_name = (
        f"{args.batch_size}_batch_{torch.cuda.device_count()}_GPU"
    )
    args.output_dir = os.path.join(args.output_dir, setting_name)
    results_file = os.path.join(args.output_dir, "results.json")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = root_logger(os.path.join(args.output_dir, f"log_{args.encoder_name}_{args.ordered_cl_tasks}_{args.optimizer_mode}_{args.do_single}_{args.seed}_{args.comm_rounds}_{args.local_epochs}.txt"))
    logger.info("Arguments: %s", args)

    if args.optimizer_mode == "adapter":
        assert args.adapter_reduction_factor > 0
    if args.optimizer_mode == "freeze_bottom_k_layers":
        assert args.layers_to_freeze > 0

    # --------------------- Load the ContinualLearner model, based on encoder_name argument  ---------------------
    model_config = model_configs[args.encoder_name]  # encoder_name is ViLT

    if args.debug:
        for k, _ in task_configs.items():
            args.num_epochs = 1

    args.cl_algorithm = 'sequential_ft'

    if args.ordered_cl_tasks == "scene":
        args.ordered_cl_tasks = ["clove_scene_a", "clove_scene_b", "clove_scene_c",
                                "clove_scene_d", "clove_scene_e", "clove_scene_f",]
    elif args.ordered_cl_tasks == "function":
        args.ordered_cl_tasks = ["clove_function_a", "clove_function_b", "clove_function_c",
                                "clove_function_d", "clove_function_e",]
    elif args.ordered_cl_tasks == "domain":
        args.ordered_cl_tasks = [ "art",  "abstract", "vizwiz", "toronto",  "gqa",]

    model = prepare_model(args, logger)
    find_unused_parameters = "vilt" in args.encoder_name or "dat" in args.optimizer_mode
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters, broadcast_buffers=False)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # ------------------------------------------ Print some model info ------------------------------------------
    logger.info("Succesfully initialized {}-based Continual Learner".format(model_config["encoder_name"]))
    logger.info("{} task heads: {}".format(len(args.ordered_cl_tasks), ",".join(args.ordered_cl_tasks)))
    logger.info("CL Algorithm: {}".format(args.cl_algorithm))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total Parameters: {:.2f}M".format(total_params * 10 ** -6))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    logger.info("Trainable Parameters: {:.2f}M ({:.2f}%)".format(trainable_params * 10 ** -6, (trainable_params / total_params * 100)))
    logger.info("Model checkpoints sa+ved to {}".format(args.output_dir))
    logger.info("-" * 80)

    if args.do_train:
        # If continuing an already-started experiment, and old upstream results were detected, then load the old results from json
        results = []
        # if os.path.isfile(results_file):
        #     results = json.load(open(results_file))
        #     logger.info("-" * 80)
        #     logger.info("Cached results:")
        #     for i, r in enumerate(results):
        #         task_key = r["task_key"]
        #         best_score = r["best_score"]
        #         logger.info("Task #{}: {} - best score = {:.2f}".format(i + 1, task_configs[task_key]["task_name"], best_score))

        # Begin training on VL tasks sequentially
        logger.info("-" * 80)
        logger.info("Training models on Vision-Language continual learning tasks...")

        # answer_list = []
        # for task_num, task_key in enumerate(args.ordered_cl_tasks):
        #     if 'albef' in args.encoder_name:
        #         task_trainer_class = task_configs[task_key]["task_trainer"]  # VQATrainer
        #         task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir=None, accelerator=accelerator)  # into VQATrainer in train_vqa.py
        #         answer_list += task_trainer.vqa_train_dataloader.dataset.answer_list
        # answer_list = list(set(answer_list))

        if args.do_single:
            # Single Task
            for task_num, task_key in enumerate(args.ordered_cl_tasks):
                for comm_round in range(args.comm_rounds):
                    logger.info("-" * 80)
                    task_name = task_configs[task_key]["task_name"]
                    task_output_dir = os.path.join(args.output_dir, "checkpoints", "task{}_{}".format(task_num, task_key))

                    # Create the Trainer method for the current CL task, and call the train method
                    logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num + 1, task_name))
                    task_trainer_class = task_configs[task_key]["task_trainer"]  # VQATrainer
                    task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)  # into VQATrainer in train_vqa.py
                    _, _ = task_trainer.train(model)

                    accelerator.wait_for_everyone()
                    torch.cuda.empty_cache()
                    accelerator.free_memory()

                with torch.no_grad():
                    eval_score_server = task_trainer.eval(model)
                    if isinstance(eval_score_server, list):
                        logger.info("{} test score server = {}".format(task_name, eval_score_server))
                    else:
                        logger.info("{} test score server = {:.2f}".format(task_name, eval_score_server))

                del model, task_trainer
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
                accelerator.free_memory()

                model = prepare_model(args, logger)
                find_unused_parameters = "vilt" in args.encoder_name or "dat" in args.optimizer_mode
                ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters, broadcast_buffers=False)
                accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
                device = accelerator.device

        else:
            # store the personalized parameters
            if hasattr(args, 'personal_params_names'):
                personal_params = {}
                print('#####################################')
                print('Personalized parameters:')
                for task_num, task_key in enumerate(args.ordered_cl_tasks):
                    personal_params[task_key] = {}
                    for n in model.state_dict().keys():
                        for pn in args.personal_params_names:
                            if pn in n:
                                if task_num==0 and torch.distributed.get_rank()==0: print(n)
                                personal_params[task_key][n] = model.state_dict()[n].data.clone().detach()


            for comm_round in range(args.comm_rounds):
                c_models = []
                nums = [1 for _ in range(len(args.ordered_cl_tasks))]
                # if ('v9' in args.optimizer_mode or 'v10' in args.optimizer_mode) and comm_round == args.comm_rounds - 1:
                #     model.albef_model.albef.set_active_gating()
                #     model.albef_model.albef.zero_grad()
                #     for n, p in model.named_parameters():
                #         if 'adapter' in n:
                #             if 'gating' in n:
                #                 p.requires_grad = True
                #             else:
                #                 p.requires_grad = False

                for task_num, task_key in enumerate(args.ordered_cl_tasks):
                    logger.info("-" * 80)
                    task_name = task_configs[task_key]["task_name"]
                    task_output_dir = os.path.join(args.output_dir, "checkpoints", "task{}_{}".format(task_num, task_key))

                    # Load the personalized parameters
                    temp_model = copy.deepcopy(model)
                    if hasattr(args, 'personal_params_names'):
                        for n in temp_model.state_dict().keys():
                            for pn in args.personal_params_names:
                                if pn in n:
                                    param = personal_params[task_key][n]
                                    temp_model.state_dict()[n].data.copy_(param)

                    # Create the Trainer method for the current CL task, and call the train method
                    logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num + 1, task_name))
                    task_trainer_class = task_configs[task_key]["task_trainer"]  # VQATrainer
                    task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)  # into VQATrainer in train_vqa.py

                    _, c_model = task_trainer.train(temp_model, comm_round)

                    # logger.info("Best {} evaluation score = {:.2f}".format(task_name, best_eval_score))
                    accelerator.wait_for_everyone()
                    torch.cuda.empty_cache()
                    accelerator.free_memory()

                    # Store the personalized parameters
                    if hasattr(args, 'personal_params_names'):
                        for n in c_model.state_dict().keys():
                            for pn in args.personal_params_names:
                                if pn in n:
                                    personal_params[task_key][n] = c_model.state_dict()[n].data.clone().detach()

                    c_model_dict = {}
                    for n in c_model.state_dict().keys():
                        if n in model.comm_state_dict_names:
                            c_model_dict[n] = c_model.state_dict()[n].data.to(device)
                    c_models.append(c_model_dict)
                    del task_trainer, c_model, temp_model

                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
                accelerator.free_memory()

                model = get_average_net(model, c_models, nums, args.ordered_cl_tasks, device)
                if not os.path.isdir(task_output_dir):
                    os.makedirs(task_output_dir, exist_ok=True)
                del c_models
                model.to(device)

                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
                accelerator.free_memory()

                if comm_round % 5 == 0 or args.comm_rounds - 1 == comm_round:
                    scores = 0.
                    scores_test_server = 0.
                    scores_test_local = 0.
                    with torch.no_grad():
                        for task_num, task_key in enumerate(args.ordered_cl_tasks):
                            task_name = task_configs[task_key]["task_name"]
                            # logger.info("Training {} model on task #{}: {}".format(args.encoder_name, task_num + 1, task_name))
                            task_output_dir = os.path.join(args.output_dir, "checkpoints", "task{}_{}".format(task_num, task_key))
                            task_trainer_class = task_configs[task_key]["task_trainer"]  # VQATrainer
                            task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)  # into VQATrainer in train_vqa.py
                            # Load the personalized parameters
                            if hasattr(args, 'personal_params_names'):
                                for n in model.state_dict().keys():
                                    for pn in args.personal_params_names:
                                        if pn in n:
                                            param = personal_params[task_key][n]
                                            model.state_dict()[n].data.copy_(param)

                            # eval_score_server = task_trainer.eval(model)
                            # logger.info("{} eval score server = {:.2f}".format(task_name, eval_score_server))
                            # scores += eval_score_server
                            # accelerator.wait_for_everyone()
                            # torch.cuda.empty_cache()
                            # accelerator.free_memory()

                            eval_score_server = task_trainer.eval(model)
                            if isinstance(eval_score_server, list):
                                logger.info("{} test score server = {}".format(task_name, eval_score_server))
                            else:
                                logger.info("{} test score server = {:.2f}".format(task_name, eval_score_server))
                                scores_test_server += eval_score_server
                            del task_trainer
                            accelerator.wait_for_everyone()
                            torch.cuda.empty_cache()
                            accelerator.free_memory()

                        # logger.info("Round {}: Avg eval score server = {:.2f}".format(comm_round, scores / len(args.ordered_cl_tasks)))
                        logger.info("Round {}: Avg test score server = {:.2f}".format(comm_round, scores_test_server / len(args.ordered_cl_tasks)))

if __name__ == "__main__":
    main()

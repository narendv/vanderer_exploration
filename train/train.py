# Ignore FutureWarning
# import warnings
# from pydantic.warnings import UnsupportedFieldAttributeWarning
# warnings.simplefilter("error", UnsupportedFieldAttributeWarning)
# The pydantic library emits UnsupportedFieldAttributeWarning in some setups; treat as ignore to avoid import-time errors
# warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

from vanderer_paths import expand_paths
import os
import sys
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

# Add the diffusion_policy submodule to Python path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    # Distributed training setup
    use_distributed = os.environ.get("RANK") is not None
    if use_distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if rank == 0:
            print(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if "gpu_ids" not in config:
                config["gpu_ids"] = [0]
            elif config["gpu_ids"] == "auto":
                config["gpu_ids"] = list(range(torch.cuda.device_count()))
            elif type(config["gpu_ids"]) == int:
                config["gpu_ids"] = [config["gpu_ids"]]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(x) for x in config["gpu_ids"]]
            )
            print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            print("Using cpu")

        first_gpu_id = config["gpu_ids"][0]
        device = torch.device(
            f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
        )

    # Rank-aware logging helper
    def log(msg):
        if rank == 0:
            print(msg)

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        data_waypoint_spacing = data_config.get("waypoint_spacing", 1)
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                        encoder_type=config["obs_encoder"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    # Create sampler for distributed training
    train_sampler = None
    shuffle_train = True
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle_train = False  # Sampler handles shuffling

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_sampler = None
        if use_distributed:
            test_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False if test_sampler else True,
            sampler=test_sampler,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                obs_encoder=config["obs_encoder"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                use_cache_encodings=True,
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=3 if config["learn_angle"] else 2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
        ema_model = EMAModel(parameters=model.parameters(),power=0.75)
        model = model.to(device)
        ema_model.to(device)
        # Note: torch.compile is applied after DataParallel/DDP wrapping below
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        log(f"Clipping gradients to {config['max_norm']}")
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    scheduler_name = config.get("scheduler")
    if isinstance(scheduler_name, str):
        scheduler_name = scheduler_name.lower().strip()
    if scheduler_name in [None, "none", "null", "off", "false"]:
        log("No LR scheduler")
    else:
        config["scheduler"] = scheduler_name
        if config["scheduler"] == "cosine":
            log(f"Using cosine annealing with T_max {config['epochs']}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            log(f"Using cyclic LR with cycle {config['cyclic_period']}")
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "onecycle":
            log(f"Using OneCycleLR with max_lr {lr} and total_steps {config['epochs']*len(train_loader)}")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                pct_start=config["warmup_fraction"],
                total_steps=config["epochs"]*len(train_loader),
                anneal_strategy='cos',
            )
        elif config["scheduler"] == "plateau":
            log("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            log("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        log("Loading model from " + load_project_folder)
        latest_path = os.path.join(load_project_folder, "orig_latest.pth")
        latest_ema_path = os.path.join(load_project_folder, "ema_latest.pth")
        latest_checkpoint = torch.load(latest_path, map_location=device)
        load_model(model, config["model_type"], latest_checkpoint)
        ema_state_dict = torch.load(latest_ema_path, map_location=device, weights_only=True)
        ema_model.load_state_dict(ema_state_dict)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        elif "load_epoch" in config:
            current_epoch = config["load_epoch"] + 1
        print(f"Resuming from epoch {current_epoch}")

    # Move model to device BEFORE wrapping with DDP
    model = model.to(device)

    # Multi-GPU / Distributed training
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Handle cases where not all parameters are used in forward
        )
        # Set static graph to handle parameter reuse in checkpoint/ema
        model._set_static_graph()
    elif len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])

    # # Apply torch.compile after DataParallel/DDP wrapping to preserve attribute access
    # if config["model_type"] == "nomad":
    #     model = torch.compile(model)

    if "load_run" in config and not config["finetune"]:  # load optimizer and scheduler only if resuming training, not if finetuning
        optimizer_path = os.path.join(load_project_folder, "optimizer_latest.pth")
        if os.path.exists(optimizer_path):
            optimizer_checkpoint = torch.load(optimizer_path, weights_only=False)
            optimizer.load_state_dict(optimizer_checkpoint)
        scheduler_path = os.path.join(load_project_folder, "scheduler_latest.pth")
        if os.path.exists(scheduler_path) and scheduler is not None:
            scheduler_checkpoint = torch.load(scheduler_path, weights_only=False)
            scheduler.load_state_dict(scheduler_checkpoint)

    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            waypoint_spacing=data_waypoint_spacing,
        )

    # Cleanup distributed training
    if use_distributed:
        torch.distributed.destroy_process_group()

    log("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = expand_paths(yaml.safe_load(f))

    config.update(user_config)

    if config['finetune']:
        if len(config["datasets"]) == 1:
            dataset_name = list(config["datasets"].keys())[0]
            config["run_name"] += '_' +  dataset_name
        else:
            config["run_name"] += '_all_towns'
    else:
        config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
        
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    
    # Only create directories on rank 0 to avoid race conditions in distributed training
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(
            config[
                "project_folder"
            ],  # should error if dir already exists to avoid overwriting and old project
        )

    # Only initialize wandb on rank 0 to avoid duplicate runs
    if config["use_wandb"] and int(os.environ.get("RANK", 0)) == 0:
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    # Only print config on rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        print(config)
    main(config)

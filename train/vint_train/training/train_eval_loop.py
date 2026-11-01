from copy import deepcopy
import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad, use_ema_weights, synchronize_loss_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        
        # Synchronize average test loss across all ranks
        if use_wandb:
            avg_loss_tensor = torch.tensor(np.mean(avg_total_test_loss), device=device)
            loss_dict = {"avg_total_test_loss": avg_loss_tensor}
            sync_losses = synchronize_loss_dict(loss_dict)
            
            # Only rank 0 saves checkpoints and logs
            if int(os.environ.get("RANK", 0)) == 0:
                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(checkpoint, latest_path)
                torch.save(checkpoint, numbered_path)  # keep track of model at every epoch
                
                # log average eval loss
                wandb.log({}, commit=False)

                if scheduler is not None:
                    # scheduler calls based on the type of scheduler
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(np.mean(avg_total_test_loss))
                    else:
                        scheduler.step()
                wandb.log({
                    "avg_total_test_loss": sync_losses["avg_total_test_loss"],
                    "lr": optimizer.param_groups[0]["lr"],
                }, commit=False)
        else:
            # Save checkpoint even without wandb
            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(checkpoint, latest_path)
            torch.save(checkpoint, numbered_path)
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(np.mean(avg_total_test_loss))
                else:
                    scheduler.step()

    # Flush the last set of eval logs (only rank 0)
    if use_wandb and int(os.environ.get("RANK", 0)) == 0:
        wandb.log({})
    print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    waypoint_spacing: int = 1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    
    
    for epoch in range(current_epoch, epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{epochs - 1}"
            )
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )

        if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
            # numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            # torch.save(ema_model.state_dict(), numbered_path)
            latest_path = os.path.join(project_folder, f"ema_latest.pth")
            torch.save(ema_model.state_dict(), latest_path)
            print(f"Saved EMA model to {latest_path}")

            # numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            # torch.save(model.state_dict(), numbered_path)
            latest_path = os.path.join(project_folder, f"orig_latest.pth")
            
            # Extract state_dict properly (remove "module." prefix if wrapped in DataParallel/DDP)
            model_state = model.state_dict()
            if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                # If model is wrapped, use .module to get the underlying model
                model_state = model.module.state_dict()
            
            torch.save(model_state, latest_path)
            print(f"Saved model to {latest_path}")

            # save optimizer
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            if lr_scheduler is not None:
                # save scheduler
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0:
            with torch.no_grad():
                for dataset_type in test_dataloaders:
                    print(
                        f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{epochs - 1}"
                    )
                    loader = test_dataloaders[dataset_type]
                    evaluate_nomad(
                        eval_type=dataset_type,
                        model=model,
                        ema_model=ema_model,
                        dataloader=loader,
                        transform=transform,
                        device=device,
                        noise_scheduler=noise_scheduler,
                        goal_mask_prob=goal_mask_prob,
                        project_folder=project_folder,
                        epoch=epoch,
                        print_log_freq=print_log_freq,
                        num_images_log=num_images_log,
                        wandb_log_freq=wandb_log_freq,
                        use_wandb=use_wandb,
                        eval_fraction=eval_fraction,
                        viz_type="image",
                        waypoint_spacing=waypoint_spacing
                    )
        
        # Synchronize learning rate across all ranks and log on rank 0
        if use_wandb:
            lr_tensor = torch.tensor(optimizer.param_groups[0]["lr"], device=device)
            loss_dict = {"lr": lr_tensor}
            sync_losses = synchronize_loss_dict(loss_dict)
            
            if int(os.environ.get("RANK", 0)) == 0:
                wandb.log({
                    "lr": sync_losses["lr"],
                }, commit=False)
                wandb.log({}, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()
            
    # Flush the last set of eval logs (only rank 0)
    if use_wandb and int(os.environ.get("RANK", 0)) == 0:
        wandb.log({})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        if "_orig_mod.module." in list(state_dict.keys())[0]:
            prefix = "_orig_mod.module."
        elif "_orig_mod." in list(state_dict.keys())[0]:
            prefix = "_orig_mod."
        elif "module." in list(state_dict.keys())[0]:
            prefix = "module."

        if prefix is not None:
            new_checkpoint = {}
            for k, v in state_dict.items():
                new_key = k.replace(prefix, "")
                new_checkpoint[new_key] = v
            checkpoint = new_checkpoint
        model.load_state_dict(checkpoint, strict=True)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=True)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=True)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

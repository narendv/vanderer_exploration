#!/usr/bin/env python3
"""
Fast per-frame embedding preprocessor.

- Scans all trajectories and timesteps
- Loads each image independently (no context windows)
- Encodes in large GPU batches with AMP
- Writes compact float16 embeddings into a single LMDB
- Keys are "{traj_name}_{timestep}"

Requirements:
  - efficientnet_pytorch
  - lmdb
  - tqdm
  - torchvision
  - easydict
  - Your project utils: replace_bn_with_gn, img_path_to_data, get_data_path
"""

import os
import io
import sys
import yaml
import math
import lmdb
import torch
import pickle
import random
import argparse
import tqdm
from pathlib import Path
from typing import Any, Dict, List, Tuple
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from efficientnet_pytorch import EfficientNet
from diffusers.models import AutoencoderKL

# Add parent directory for project imports

from vint_train.models.nomad.nomad_vint import replace_bn_with_gn
from vint_train.data.data_utils import img_path_to_data, get_data_path


# ---------------------------
# Index building
# ---------------------------

def load_traj_names(split_dir: str) -> List[str]:
    fname = os.path.join(split_dir, "traj_names.txt")
    with open(fname, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_traj_len(data_dir: str, traj_name: str) -> int:
    """Reads traj_data.pkl to estimate number of frames."""
    pkl_path = os.path.join(data_dir, traj_name, "traj_data.pkl")
    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)
    # conservative: len of positions is a good proxy for frames
    return len(traj["position"])


def build_index(
    data_dir: str,
    split_dir: str,
    min_sequence_length: int = 1,
) -> List[Tuple[str, int]]:
    """
    Build a flat index of all (traj_name, timestep) to process.
    """
    trajs = load_traj_names(split_dir)
    items: List[Tuple[str, int]] = []
    for traj in tqdm.tqdm(trajs, desc="Indexing"):
        try:
            T = read_traj_len(data_dir, traj)
            if T < min_sequence_length:
                continue
            for t in range(T):
                items.append((traj, t))
        except Exception as e:
            print(f"[WARN] Skipping {traj}: {e}")
            continue
    print(f"Total frames to encode: {len(items)}")
    return items


# ---------------------------
# Dataset
# ---------------------------

class FrameDataset(Dataset):
    def __init__(
        self,
        index: List[Tuple[str, int]],
        data_dir: str,
        image_size: Tuple[int, int],
        encoder_name: str,
    ):
        self.index = index
        self.data_dir = data_dir
        self.image_size = image_size
        self.encoder_name = encoder_name

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        traj_name, t = self.index[i]
        img_path = get_data_path(self.data_dir, traj_name, t)
        img_tensor = img_path_to_data(img_path, self.image_size)  # C,H,W float[0..1]
        if self.encoder_name == 'nomad':
            img_tensor = TF.normalize(
                img_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        key = f"{traj_name}_{t}"
        return img_tensor, key


def collate_fn(batch):
    # batch: List[(C,H,W tensor, key)]
    imgs = torch.stack([x[0] for x in batch], dim=0)  # B,C,H,W
    keys = [x[1] for x in batch]
    return imgs, keys


# ---------------------------
# Encoder setup
# ---------------------------

def load_encoder(checkpoint_path: str, device: str) -> torch.nn.Module:
    enc = EfficientNet.from_name("efficientnet-b0", in_channels=3)
    enc = replace_bn_with_gn(enc)

    print(f"Loading encoder weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # If checkpoint might be a PL checkpoint with 'state_dict'
    state = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]

    enc_state = {}
    for k, v in state.items():
        if k.startswith("vision_encoder.obs_encoder."):
            new_k = k[len("vision_encoder.obs_encoder."):]
            enc_state[new_k] = v

    missing, unexpected = enc.load_state_dict(enc_state, strict=True)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

    enc.to(device).eval()
    return enc

def load_encoder_kl(checkpoint_path: str, device: str) -> torch.nn.Module:
    enc = AutoencoderKL.from_pretrained(checkpoint_path)
    enc.requires_grad_(False) # freeze VAE weights
    enc.to(device).eval()
    return enc


@torch.inference_mode()
def encode_batch_effnet(enc: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) normalized
    returns: (B,D) float32 features
    """
    with torch.no_grad():
        feats = enc.extract_features(x)
        feats = enc._avg_pooling(feats)
        # include_top False in EfficientNet.from_name by default, we just pool+flatten
        feats = torch.flatten(feats, 1)
    return feats

@torch.inference_mode()
def encode_batch_kl(enc: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) in [0,1] range (will be converted to [-1,1] for AutoencoderKL)
    returns: (B,C,H/8,W/8) latent features
    """
    with torch.no_grad():
        # AutoencoderKL expects inputs in [-1, 1] range
        x_scaled = x * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
        
        # Encode images to latents
        latents = enc.encode(x_scaled).latent_dist.mean  # (B, C, H/8, W/8)
        
        # Optional: flatten spatial dimensions if you need 1D features
        # B, C, H, W = latents.shape
        # feats = latents.view(B, C * H * W)  # (B, D)
        # return feats
        
        return latents  # Keep spatial structure: (B, 4, H/8, W/8)


def encode_batch(enc: torch.nn.Module, x: torch.Tensor, encoder_type: str) -> torch.Tensor:
    """
    Generic encode batch function that dispatches to the correct encoder.
    """
    if encoder_type == "efficientnet":
        return encode_batch_effnet(enc, x)
    elif encoder_type == "kl":
        return encode_batch_kl(enc, x)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ---------------------------
# LMDB writer
# ---------------------------

class LMDBWriter:
    def __init__(self, path: str, map_size_bytes: int = 20 * 1024**3, readahead: bool = False):
        if os.path.islink(path):
            raise ValueError(f"{path} links to shared data; remove the link before regenerating it.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.env = lmdb.open(
            path,
            map_size=map_size_bytes,
            subdir=True,
            lock=True,
            readahead=readahead,
            meminit=False,
        )
        self.txn = self.env.begin(write=True)
        self.count_since_commit = 0

    def put_tensor(self, key_str: str, tensor: torch.Tensor):
        # Store as float16 to save space
        t = tensor.detach().cpu().to(torch.float16).contiguous()
        buf = io.BytesIO()
        torch.save(t, buf)
        self.txn.put(key_str.encode("utf-8"), buf.getvalue())
        self.count_since_commit += 1

    def maybe_commit(self, commit_interval: int = 10000):
        if self.count_since_commit >= commit_interval:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.count_since_commit = 0

    def close(self):
        self.txn.commit()
        self.env.sync()
        self.env.close()


# ---------------------------
# Main processing
# ---------------------------

def preprocess_all_frames(
    encoder_name: str,
    encoder_checkpoint: str,
    data_dir: str,
    split_dir: str,
    out_lmdb_path: str,
    image_size: Tuple[int, int] = (96, 96),
    normalize: bool = True,
    min_sequence_length: int = 1,
    batch_size: int = 512,
    num_workers: int = 8,
    commit_interval: int = 10000,
    seed: int = 42,
):

    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if encoder_name == "efficientnet":
        enc = load_encoder(encoder_checkpoint, device)
    elif encoder_name == "kl":
        enc = load_encoder_kl(encoder_checkpoint, device)
    else:
        raise ValueError(f"Unknown encoder name: {encoder_name}")
    # from torchinfo import summary
    # summary(enc, input_size=(2, 3, image_size[0], image_size[1]))

    # Build flat index of (traj, t)
    index = build_index(data_dir, split_dir, min_sequence_length=min_sequence_length)
    if len(index) == 0:
        print("No frames to process. Exiting.")
        return

    # Dataset / Loader
    dset = FrameDataset(index, data_dir=data_dir, image_size=image_size, encoder_name=encoder_name)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    # LMDB writer
    writer = LMDBWriter(out_lmdb_path, map_size_bytes=20 * 1024**3)

    # Process
    pbar = tqdm.tqdm(loader, total=math.ceil(len(dset) / batch_size), desc="Encoding")
    for imgs, keys in pbar:
        imgs = imgs.to(device, non_blocking=True)
        feats = encode_batch(enc, imgs, encoder_name)  # (B, D) or (B, C, H, W)
        # write each item
        for i, key in enumerate(keys):
            writer.put_tensor(key, feats[i])
        writer.maybe_commit(commit_interval=commit_interval)
        pbar.set_postfix({"last_key": keys[-1]})

    writer.close()
    print(f"✅ Embedding cache built at: {out_lmdb_path}")


def get_embedding_from_lmdb(lmdb_path: str, traj_name: str, timestep: int):
    """
    Fetch one embedding from LMDB.

    Args:
        lmdb_path: path to the LMDB file
        traj_name: trajectory name (string)
        timestep: timestep index (int)

    Returns:
        torch.Tensor with the embedding (float16 or float32 depending on how you stored it)
    """
    key = f"{traj_name}_{timestep}".encode("utf-8")

    # Open LMDB in read-only mode
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=False) as txn:
        buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Key {traj_name}_{timestep} not found in {lmdb_path}")

        # Deserialize tensor
        embedding = torch.load(io.BytesIO(buf), map_location="cpu")

    env.close()
    return embedding

def parse_args():
    ap = argparse.ArgumentParser(description="Fast per-frame embedding preprocessing")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to data folder (e.g., <data_root>/recon)")
    ap.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., recon, scand)")
    ap.add_argument("--data_split_folder", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vint_train", "data", "data_splits"),
                    help="Base path to data splits folder (dataset name will be appended)")
    ap.add_argument("--enc_checkpoint", type=str, required=True,
                    help="Encoder checkpoint path (e.g., stabilityai/sd-vae-ft-ema or path to nomad.pth)")
    ap.add_argument("--split", type=str, default="train", help="train/test")
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224], 
                    help="Image size as width height (default: 224 224)")
    ap.add_argument("--min_sequence_length", type=int, default=1,
                    help="Minimum sequence length to include (default: 1)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--commit_interval", type=int, default=10000)
    return ap.parse_args()

# python preprocess_embeddings.py --data_dir <data_root>/recon --dataset_name recon --enc_checkpoint stabilityai/sd-vae-ft-ema --split train --batch_size 64 
def main():
    args = parse_args()
    
    # Determine encoder type from checkpoint path
    if "nomad" in args.enc_checkpoint:
        encoder_name = "efficientnet"
    elif "vae" in args.enc_checkpoint:
        encoder_name = "kl"
    else:
        raise ValueError(f"Cannot determine encoder type from checkpoint: {args.enc_checkpoint}")
    
    # Construct split directory path
    split_dir = os.path.join(args.data_split_folder, args.dataset_name, args.split)
    
    # Output LMDB path
    out_lmdb = os.path.join(split_dir, f"dataset_{args.dataset_name}_{encoder_name}_encodings_{args.image_size[0]}.lmdb")

    print(f"Preprocessing embeddings:")
    print(f"  Data folder: {args.data_dir}")
    print(f"  Dataset name: {args.dataset_name}")
    print(f"  Split directory: {split_dir}")
    print(f"  Encoder: {encoder_name} ({args.enc_checkpoint})")
    print(f"  Output LMDB: {out_lmdb}")
    print(f"  Image size: {args.image_size}")
    print(f"  Min sequence length: {args.min_sequence_length}")

    preprocess_all_frames(
        encoder_name=encoder_name,
        encoder_checkpoint=args.enc_checkpoint,
        data_dir=args.data_dir,
        split_dir=split_dir,
        out_lmdb_path=out_lmdb,
        image_size=tuple(args.image_size),
        min_sequence_length=args.min_sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        commit_interval=args.commit_interval,
    )

if __name__ == "__main__":
    main()
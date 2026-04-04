"""Training loop for Loritz et al. (2024) replication.

Faithful port of their Cell 15-18: model init, optimizer setup,
training with noise injection, gradient clipping, and model saving.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from .config import FEATURES_WITHOUT_ONEHOT, PLANT_LIST, NetworkParams, ReplicationConfig
from .dataset import SequenceDataset
from .evaluator import compute_kge, compute_mae, compute_nse
from .model import RNN_network, init_forget_gate_bias


def build_dataloader(
    all_data: Dict[str, pd.DataFrame],
    plant_ids: List[str],
    all_columns: List[str],
    target: str,
    features: List[str],
    params: NetworkParams,
    standardize: bool,
    all_mean: pd.Series,
    all_std: pd.Series,
    sapf_mean: float,
    sapf_std: float,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader from a list of plant IDs (ports their Cell 13-14)."""
    datasets = []
    for plant_id in plant_ids:
        df = all_data[plant_id]
        # Reindex to all_columns to ensure consistent columns (NaN -> 0)
        df = pd.DataFrame(df, columns=all_columns)

        ds = SequenceDataset(
            df_pl=df,
            target_var=target,
            features_var=features,
            sequence_length=params.sequence_length,
            standardize=standardize,
            all_mean=all_mean,
            all_std=all_std,
            sapf_mean=sapf_mean,
            sapf_std=sapf_std,
        )
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=params.batch_size,
        shuffle=shuffle,
        drop_last=params.drop_last if shuffle else False,
    )


def train_one_epoch(
    network: RNN_network,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    min_obs: float,
    noise_std: float = 0.05,
    grad_clip: bool = True,
    max_norm: float = 1.0,
) -> float:
    """Train one epoch (ports their Cell 18 training loop).

    Key behaviors matching their code:
    - Filter out samples where target <= min_obs (zero sap flow in z-space)
    - Add 5% Gaussian noise to both input and target
    - Gradient clipping AFTER optimizer.step() (matches their code order)
    """
    network.train()
    all_train_loss = []

    for batch, target_loop in train_loader:
        # Filter zero sap flow values
        mask = target_loop > min_obs
        batch = batch[mask]
        target_loop = target_loop[mask]

        if len(batch) == 0:
            continue

        # Add noise to input and target
        noise_x = noise_std * torch.randn_like(batch)
        batch_noise = batch + noise_x
        noise_y = noise_std * torch.randn_like(target_loop)
        target_noise = target_loop + noise_y

        pred = network(batch_noise.to(device).float())
        loss = criterion(torch.squeeze(pred), target_noise.to(device).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gradient clipping AFTER step (matches their code)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_norm)

        all_train_loss.append(loss.item())

        del batch, target_loop
        torch.cuda.empty_cache()

    return np.mean(all_train_loss) if all_train_loss else float("nan")


def validate_one_epoch(
    network: RNN_network,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    min_obs: float,
) -> float:
    """Validate one epoch (ports their Cell 18 validation loop).

    Note: Their code does NOT use torch.no_grad() or network.set_to_inference()
    during validation within the training loop. We replicate this behavior.
    """
    all_val_loss = []

    for val_batch, val_target in val_loader:
        mask = val_target > min_obs
        val_batch = val_batch[mask]
        val_target = val_target[mask]

        if len(val_batch) == 0:
            continue

        val_pred = network(val_batch.to(device).float())
        val_loss = criterion(torch.squeeze(val_pred), val_target.to(device).float())

        all_val_loss.append(val_loss.item())

        del val_batch, val_target
        torch.cuda.empty_cache()

    return np.mean(all_val_loss) if all_val_loss else float("nan")


def train_model(
    network: RNN_network,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    params: NetworkParams,
    min_obs: float,
    save_dir: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """Full training loop for N epochs (ports their Cell 15-18).

    Returns:
        history: dict with 'train_loss' and 'val_loss' lists
    """
    # Initialize forget gate bias (Cell 15)
    init_forget_gate_bias(network, params.set_forget_gate)

    # Optimizer (Cell 16)
    optimizer = optim.Adam(
        network.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    criterion = nn.MSELoss(reduction="mean")
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(params.no_of_epochs):
        train_loss = train_one_epoch(
            network, train_loader, optimizer, criterion, device,
            min_obs, params.noise_std, params.grad_clip, params.max_norm,
        )
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = validate_one_epoch(
                network, val_loader, criterion, device, min_obs,
            )
        history["val_loss"].append(val_loss)

        scheduler.step()

        print(f"epoch: {epoch}")
        print(f"train_MSE: {train_loss}")
        print(f"Learning rate: {round(optimizer.param_groups[0]['lr'], 5)}")
        print(f"val_MSE: {val_loss}")

        # Save model each epoch (Cell 18)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            path_model = save_dir / f"trial_and_error_{epoch}"
            torch.save(network.state_dict(), path_model)
            print("model saved")

        print()

    print("fini")
    network.training = False  # Switch to inference mode
    return history


def predict(
    network: RNN_network,
    all_data: Dict[str, pd.DataFrame],
    test_plants: List[str],
    all_columns: List[str],
    features: List[str],
    params: NetworkParams,
    device: torch.device,
    all_mean: pd.Series,
    all_std: pd.Series,
    sapf_mean: float,
    sapf_std: float,
    plant_list: List[str],
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], Dict[str, int]]:
    """Run prediction on test data (ports their Cell 20).

    Key behaviors:
    - Test data NOT standardized for target (standardize=False)
    - Predictions de-normalized: pred * sapf_std + sapf_mean
    - Only non-zero observations used (targets > 0)

    Returns:
        pred_dict, obs_dict, test_counters
    """
    network.eval()
    pred_dict: Dict[str, List[np.ndarray]] = {}
    obs_dict: Dict[str, List[np.ndarray]] = {}
    test_counters = {name: 0 for name in plant_list}

    for plant_id in test_plants:
        df = all_data[plant_id]
        df = pd.DataFrame(df, columns=all_columns)

        # Detect genus from one-hot columns
        genus_cols = [c for c in plant_list if c in df.columns]
        plant_name = None
        for gc in genus_cols:
            if df.iloc[0][gc] == 1:
                plant_name = gc
                break

        if plant_name is None:
            continue

        ds = SequenceDataset(
            df_pl=df,
            target_var="sapf",
            features_var=features,
            sequence_length=params.sequence_length,
            standardize=False,  # Test target NOT standardized
            all_mean=all_mean,
            all_std=all_std,
            sapf_mean=sapf_mean,
            sapf_std=sapf_std,
        )

        test_loader = DataLoader(
            ds,
            batch_size=params.batch_size,
            shuffle=False,
            drop_last=False,
        )

        for batch, targets in test_loader:
            # Only use non-zero observations
            mask = targets > 0
            if mask.sum() == 0:
                continue

            y_pred = network(batch[mask].to(device).float())
            y_pred = y_pred.flatten().cpu().detach().numpy()
            # De-normalize predictions
            y_pred = y_pred * sapf_std + sapf_mean

            y_obs = targets[mask].flatten().cpu().detach().numpy()

            if plant_name not in pred_dict:
                pred_dict[plant_name] = []
                obs_dict[plant_name] = []

            pred_dict[plant_name].append(y_pred)
            obs_dict[plant_name].append(y_obs)

            del y_pred, y_obs
            torch.cuda.empty_cache()

        test_counters[plant_name] += 1

    return pred_dict, obs_dict, test_counters

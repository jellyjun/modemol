from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from addict import Dict
from tqdm import tqdm

from src import Model
from src.dataset import get_dataloader
from src.process import get_process
from src.utils.logger import default_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def continuous_functional_contrastive_loss(
    features: torch.Tensor,
    properties: torch.Tensor,
    temperature: float,
    functional_temperature: float,
) -> torch.Tensor:
    if features.size(0) < 2:
        return features.new_tensor(0.0)
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(logits) - torch.eye(logits.size(0), device=features.device)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    functional_dist = torch.cdist(properties, properties, p=2).pow(2)
    target = torch.exp(-functional_dist / functional_temperature)
    target = target * logits_mask
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(target.detach() * log_prob).sum(dim=1).mean()


def freeze_except(model: Model, trainable_modules: set[str]) -> None:
    for name, module in model.items():
        requires_grad = name in trainable_modules
        for param in module.parameters():
            param.requires_grad = requires_grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--functional-temperature", type=float, default=0.01)
    parser.add_argument("--lambda-recon", type=float, default=0.2)
    parser.add_argument("--lambda-kl", type=float, default=0.0005)
    parser.add_argument("--lambda-contrastive", type=float, default=0.5)
    parser.add_argument("--lambda-regularization", type=float, default=0.05)
    parser.add_argument("--loss-log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--gpuid", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = default_logger(
        os.path.join(args.output_dir, "log.txt"),
        stream_level="info",
        file_level="debug",
    )
    with open(args.config, "rb") as handle:
        config = Dict(yaml.load(handle, yaml.Loader))
    with open(os.path.join(args.output_dir, "run_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, sort_keys=False)

    device = torch.device("cuda", index=args.gpuid) if torch.cuda.is_available() else torch.device("cpu")
    logger.warning(f"DEVICE: {device}")

    dl_config = copy.deepcopy(config.training.data.train)
    dl_config.datasets.dfs = Dict({"finetune": {"path": args.data_csv}})
    dl_config.datasets.datasets.molecules.df = "finetune"
    dl_config.datasets.datasets.molecules.prop_cols = ["gsk", "jnk"]
    dl_config.datasets.datasets.molecules.prop_name = "gsk_jnk"
    dl_config.num_tokens = args.num_tokens
    dl_config.max_batch_size = args.max_batch_size
    dl_config.seed = args.seed
    dl_train = get_dataloader(logger=logger, device=device, **dl_config)

    model = Model(logger=logger, **copy.deepcopy(config.model))
    model.load(path=args.weight_path, strict=True)
    model.to(device)

    base_model = Model(logger=logger, **copy.deepcopy(config.model))
    base_model.load(path=args.weight_path, strict=True)
    base_model.to(device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    trainable_modules = {"encoder", "pooler", "latent2mu", "latent2var", "vae"}
    freeze_except(model, trainable_modules)
    model.train()

    train_process_configs = [
        process for process in config.training.train_loop
        if process.get("module") not in {"property_head", "property_mse", "property_mse_factor"}
    ]
    train_processes = [get_process(**copy.deepcopy(process)) for process in train_process_configs]
    encode_process_configs = [
        {"module": "masker", "input": "input", "output": "input_padding_mask"},
        {"module": "enc_embedding", "input": "input", "output": "input_emb"},
        {"module": "encoder", "input": ["input_emb", "input_padding_mask"], "output": "memory"},
        {
            "type": "function",
            "function": {"type": "transpose", "dim0": 0, "dim1": 1},
            "input": "input_padding_mask",
            "output": "input_padding_mask2",
        },
        {"module": "pooler", "input": ["memory", "input_padding_mask2"], "output": "latent_base"},
        {"module": "latent2mu", "input": "latent_base", "output": "mu"},
    ]
    base_encode_processes = [get_process(**copy.deepcopy(process)) for process in encode_process_configs]

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    loss_log_path = Path(args.output_dir) / "train_contrastive_loss.csv"
    with loss_log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "epoch",
                "time",
                "total_loss",
                "sequencece",
                "kl",
                "contrastive",
                "regularization",
                "batch_size",
            ],
        )
        writer.writeheader()

    logger.info(f"Contrastive fine-tuning started. Loss log: {loss_log_path}")
    global_step = 0
    for epoch in range(args.epochs):
        epoch_totals = {k: 0.0 for k in ["total_loss", "sequencece", "kl", "contrastive", "regularization"]}
        epoch_n = 0
        with tqdm(dl_train, desc=f"epoch {epoch}", dynamic_ncols=True) as pbar:
            for batch in pbar:
                start = time.time()
                global_step += 1
                model.train()
                batch = model(batch, processes=train_processes)
                with torch.no_grad():
                    base_batch = {"input": batch["input"]}
                    base_batch = base_model(base_batch, processes=base_encode_processes)

                sequencece = batch["sequencece"]
                kl = batch["-d_kl"]
                contrastive = continuous_functional_contrastive_loss(
                    batch["mu"],
                    batch["gsk_jnk"].float(),
                    temperature=args.temperature,
                    functional_temperature=args.functional_temperature,
                )
                regularization = torch.mean((batch["mu"] - base_batch["mu"]) ** 2)
                total_loss = (
                    args.lambda_recon * sequencece
                    + args.lambda_kl * kl
                    + args.lambda_contrastive * contrastive
                    + args.lambda_regularization * regularization
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                bs = int(batch["batch_size"])
                epoch_n += bs
                values = {
                    "total_loss": float(total_loss.detach().cpu()),
                    "sequencece": float(sequencece.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                    "contrastive": float(contrastive.detach().cpu()),
                    "regularization": float(regularization.detach().cpu()),
                }
                for key, value in values.items():
                    epoch_totals[key] += value * bs
                if global_step % args.loss_log_interval == 0:
                    with loss_log_path.open("a", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            "step", "epoch", "time", "total_loss", "sequencece", "kl",
                            "contrastive", "regularization", "batch_size",
                        ])
                        writer.writerow({
                            "step": global_step,
                            "epoch": epoch,
                            "time": time.time() - start,
                            "batch_size": bs,
                            **values,
                        })
                    pbar.set_postfix(loss=f"{values['total_loss']:.4f}", contrast=f"{values['contrastive']:.4f}")
                del batch, base_batch, total_loss

        epoch_row = {key: value / max(epoch_n, 1) for key, value in epoch_totals.items()}
        logger.info(
            "epoch=%d total=%.6f seq=%.6f kl=%.6f contrastive=%.6f reg=%.6f"
            % (
                epoch,
                epoch_row["total_loss"],
                epoch_row["sequencece"],
                epoch_row["kl"],
                epoch_row["contrastive"],
                epoch_row["regularization"],
            )
        )
        model.save_state_dict(os.path.join(args.output_dir, "models", f"epoch_{epoch}"))

    model.save_state_dict(os.path.join(args.output_dir, "models", "final"))
    logger.info("Contrastive fine-tuning finished.")


if __name__ == "__main__":
    main()

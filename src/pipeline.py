from train_eval import train, evaluate
from modules import (set_seed,
                     get_dataloader,
                     get_dataloader_dynahate,
                     get_dataloader_sbic,
                     prim_encoder_con,
                     SupConLoss,
                     SentenceTriplet,
                     SST,
                     CamLoss,
                     Metrics,
                     HistoryTracker,
                     TrainingVisualizer)
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
import os
import re


def run_testing(model, data_main, tracker, output_path, device, ihc_test, sbic_test, dyna_test, ce_fn):
    print("\n🔍 Testing with best model...")
    model, _, _, _ = tracker.load_model(tracker.history["best"]["path"], model)

    test_cases = []
    if data_main == "ihc":
        test_cases = [
            (ihc_test, "in_data", "ihc_test"),
            (sbic_test, "sbic_test", "sbic_test"),
            (dyna_test, "dyna_test", "dyna_test")
        ]
    else:
        test_cases = [
            (sbic_test, "in_data", "sbic_test"),
            (ihc_test, "ihc_test", "ihc_test"),
            (dyna_test, "dyna_test", "dyna_test")
        ]
    for data_iter, test_name, folder_name in test_cases:
        evaluate(
            is_testing=True,
            data_iter=data_iter,
            test_name=test_name,
            f1_train=None,
            model=model,
            ce_fn=ce_fn,
            tracker=tracker,
            output_path=os.path.join(output_path, folder_name),
            device=device
        )
    tracker.history["tested"] = True
    tracker.save()
    tracker.purge_epoch_checkpoints()


def pipeline(
        data_path: str,
        method_dir: str,

        data_main: str,
        seed: int,

        encoder_name: str,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        lambda_weight: float,
        method: str,
        # triplet loss
        margin: float,
        # mine_margin: float,
        d_fn: str,
        beta: int,
        reducer: str,
        fallback: bool,
        # Contrastive
        temperature: float,
        am: float,
        a: float,
        r: float
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_main == "ihc":
        train_iter, valid_iter, ihc_test = get_dataloader(
            train_batch_size=batch_size, eval_batch_size=batch_size, w_aug="imp", seed=seed, base_data_path=data_path)
        _, _, sbic_test = get_dataloader_sbic(
            train_batch_size=batch_size, eval_batch_size=batch_size, w_aug="imp", seed=seed, base_data_path=data_path)
        _, _, dyna_test = get_dataloader_dynahate(
            train_batch_size=batch_size, eval_batch_size=batch_size, base_data_path=data_path)
    else:
        train_iter, valid_iter, sbic_test = get_dataloader_sbic(
            train_batch_size=batch_size, eval_batch_size=batch_size, w_aug="imp", seed=seed, base_data_path=data_path)
        _, _, ihc_test = get_dataloader(
            train_batch_size=batch_size, eval_batch_size=batch_size, w_aug="imp", seed=seed, base_data_path=data_path)
        _, _, dyna_test = get_dataloader_dynahate(
            train_batch_size=batch_size, eval_batch_size=batch_size, base_data_path=data_path)
    encoder_short_name = "bert" if "bert-base-uncased" in encoder_name else "hatebert"
    if method_dir is None:
        raise (ValueError(f"No such method dir{method_dir}"))
    if method == "contrastive":
        output_path = (
            f"{method_dir}/lr{learning_rate}_lam{lambda_weight}_temp{temperature}")
    elif method in ["semi-hard", "SST"]:
        if reducer in ["softmax", "adapt_softmax", "softmax_sh"]:
            output_path = (
                f"{method_dir}/lr{learning_rate}_lam{lambda_weight}_margin{margin}_b{beta}_fb{fallback}")
        else:
            output_path = (
                f"{method_dir}/lr{learning_rate}_lam{lambda_weight}_margin{margin}_fb{fallback}")
    elif method == "cam":
        output_path = (
            f"{method_dir}/lr{learning_rate}_lam{lambda_weight}_a{a}_m{am}_r{r}")
    os.makedirs(output_path, exist_ok=True)
    model = prim_encoder_con(
        hidden_size=768,
        label_size=2,
        encoder_type=encoder_name
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    ce_fn = nn.CrossEntropyLoss()
    if method == "contrastive":
        metric_fn = SupConLoss(temperature=temperature)
    if method == "cam":
        print(type(a), type(r), type(am))
        metric_fn = CamLoss(lambda_a=a, lambda_r=r, angular_margin_m=am, embedding_size=768, num_classes=2).to(device)
    elif method == "semi-hard":
        metric_fn = SentenceTriplet(
            margin=margin, reducers=reducer, use_fallback=fallback, beta=beta, d_fn=d_fn)
    elif method == "SST":
        metric_fn = SST(
            margin=margin, reducers=reducer, use_fallback=fallback, beta=beta, d_fn=d_fn)
    num_training_steps = int(len(train_iter)*num_epochs)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    metrics = Metrics()
    tracker = HistoryTracker.load(output_path)

    if tracker.history["best"]["f1_macro"] is None:
        tracker.history["best"]["f1_macro"] = -1
    start_epoch = 1
    latest_checkpoint = tracker.get_latest_checkpoint()

    if latest_checkpoint:
        match = re.search(r'epoch_?(\d+)', latest_checkpoint)
        if match:
            checkpoint_epoch = int(match.group(1))
            start_epoch = checkpoint_epoch + 1
        else:
            raise ValueError("Invalid checkpoint name format")
        model, optimizer, checkpoint_epoch, lr_scheduler = tracker.load_model(
            latest_checkpoint, model, optimizer, lr_scheduler=lr_scheduler)
        print(
            f"✅ Resuming from epoch {checkpoint_epoch} (training from {start_epoch})")
    else:
        print("⭐ No checkpoints found - starting from scratch")
        start_epoch = 1
    if start_epoch >= num_epochs:
        print(
            f"⚠️ Training already completed (epoch {start_epoch-1}/{num_epochs})")
        if tracker.history["best"]["path"] and not tracker.history.get("tested", False):
            run_testing(model, data_main, tracker, output_path, device,  # ADD THIS
                        ihc_test, sbic_test, dyna_test, ce_fn)
            print("💾 Resumed testing completed")
        elif tracker.history.get("tested", False):
            print("✅ Testing already completed")
        return  # Keep this return
    for epoch in range(start_epoch, num_epochs+1):
        print(f"\n🚀 Epoch {epoch}/{num_epochs}")
        current_f1, train_f1 = train(
            device=device,
            method=method,
            epoch=epoch,
            train_loader=train_iter,
            val_loader=valid_iter,
            model=model,
            batch_size=batch_size,
            lam=lambda_weight,
            metric_fn=metric_fn,
            ce_fn=ce_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tracker=tracker,
            metrics=metrics
        )
        if tracker.best_f1_score(epoch, current_f1, train_f1, model, optimizer):
            print(
                f"🏆 New best model at epoch {epoch} with F1: {current_f1:.4f}")
        tracker.save()
    print(f"💾 Saved checkpoint and metrics for epoch {epoch}")
    visualizer = TrainingVisualizer(tracker.history)
    visualizer.plot_metrics(output_path)
    if tracker.history["best"]["path"] and not tracker.history.get("tested", False):
        run_testing(model, data_main, tracker, output_path, device,
                    ihc_test, sbic_test, dyna_test, ce_fn)
    elif tracker.history.get("tested", False):
        print("✅ Testing already completed")

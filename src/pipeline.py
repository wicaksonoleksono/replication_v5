from train_eval import train,evaluate
from modules import (set_seed,
                     get_dataloader,
                     get_dataloader_dynahate,
                     get_dataloader_sbic,
                     prim_encoder_con,
                     SupConLoss,
                     SentenceTriplet,
                     Metrics,
                     HistoryTracker,
                     TrainingVisualizer)
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
import os
import re
def pipeline(
        data_path:str,
        output_base:str,

        data_main:str,
        seed:int,

        encoder_name:str,
        learning_rate:float,
        batch_size:int,
        num_epochs:int,
        lambda_weight:float,

        method:str,

        # triplet loss
        margin: float ,
        beta: int ,
        reducer: str ,
        fallback: bool ,
        # Contrastive
        temperature: float,
        ):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_main == "ihc":
        train_iter,valid_iter,ihc_test= get_dataloader(train_batch_size=batch_size,eval_batch_size=batch_size,w_aug="imp",seed=seed,base_data_path=data_path)
        _,_,sbic_test= get_dataloader_sbic(train_batch_size=batch_size,eval_batch_size=batch_size,w_aug="imp",seed=seed,base_data_path=data_path)
        _,_,dyna_test= get_dataloader_dynahate(train_batch_size=batch_size,eval_batch_size=batch_size,base_data_path=data_path)
    else:
        train_iter,valid_iter,ihc_test= get_dataloader_sbic(train_batch_size=batch_size,eval_batch_size=batch_size,w_aug="imp",seed=seed,base_data_path=data_path)
        _,_,ihc_test= get_dataloader(train_batch_size=batch_size,eval_batch_size=batch_size,w_aug="imp",seed=seed,base_data_path=data_path)
        _,_,dyna_test= get_dataloader_dynahate(train_batch_size=batch_size,eval_batch_size=batch_size,base_data_path=data_path)

    encoder_short_name = "bert" if "bert-base-uncased" in encoder_name else "hatebert"
    if method == "contrastive":
        output_path = (f"{output_base}.{method}/{data_main}_{encoder_short_name}_lr{learning_rate}_lam{lambda_weight}_temp{temperature}")
    elif method =="semi-hard":
        if reducer in ["softmax","adapt_softmax"]: 
            output_path = (f"{output_base}.{method}/{data_main}_{encoder_short_name}_lr{learning_rate}_lam{lambda_weight}_margin{margin}_red{reducer}_b{beta}_fb{fallback}")
        else:
            output_path = (f"{output_base}.{method}/{data_main}_{encoder_short_name}_lr{learning_rate}_lam{lambda_weight}_margin{margin}")
    os.makedirs(output_path,exist_ok=True)
    # Building model initializing losses and such . 
    model = prim_encoder_con(
        hidden_size=768,
        label_size=2,
        encoder_type=encoder_name
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    ce_fn = nn.CrossEntropyLoss()
    if method == "contrastive" :
        metric_fn = SupConLoss(temperature=temperature)
    elif method =="semi-hard": 
        metric_fn = SentenceTriplet(margin=margin,reducers=reducer,use_fallback=fallback,beta=beta)
    # lr scheduler 
    num_training_steps = int(len(train_iter)*num_epochs)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    # Building model initializing losses and such . 

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
        model, optimizer, checkpoint_epoch,lr_scheduler= tracker.load_model(latest_checkpoint,model,optimizer,lr_scheduler=lr_scheduler)
        print(f"✅ Resuming from epoch {checkpoint_epoch} (training from {start_epoch})")
    else:
        print("⭐ No checkpoints found - starting from scratch")
        start_epoch=1

    if start_epoch >= num_epochs:
        print(f"⚠️ Training already completed (epoch {start_epoch-1}/{num_epochs})")
        return
    for epoch in range(start_epoch,num_epochs+1):
        print(f"\n🚀 Epoch {epoch}/{num_epochs}")
        current_f1=train(
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
        if tracker.best_f1_score(epoch,current_f1,model,optimizer):
            print(f"🏆 New best model at epoch {epoch} with F1: {current_f1:.4f}")
        tracker.save()
    print(f"💾 Saved checkpoint and metrics for epoch {epoch}")
    visualizer = TrainingVisualizer(tracker.history)
    visualizer.plot_metrics(output_path)
    print(tracker.history)
    if tracker.history["best"]["path"]:
        print("\n🔍 Testing with best model...")
        model, _, _ = tracker.load_model(tracker.history["best"]["path"], model)
        if data_main == "ihc":
            evaluate(
                is_testing=True,
                data_iter=ihc_test,
                test_name="in_data",
                model=model,
                batch_size=batch_size,
                ce_fn=ce_fn,
                tracker=tracker,
                output_path=os.path.join(output_path, "ihc_test/"),
                device=device 
            )
            evaluate(
                is_testing=True,
                data_iter=sbic_test,
                test_name="sbic_test",
                model=model,
                batch_size=batch_size,
                ce_fn=ce_fn,
                tracker=tracker,
                output_path=os.path.join(output_path, "sbic_test/"),
                device=device
            )
            evaluate(
                is_testing=True,
                data_iter=dyna_test,
                test_name="dyna_test",
                model=model,
                batch_size=batch_size,
                ce_fn=ce_fn,
                tracker=tracker,
                output_path=os.path.join(output_path, "dyna_test/"),
                device=device
            )
        else:  
            evaluate(
                is_testing=True,
                data_iter=sbic_test,
                test_name="in_data",
                model=model,
                batch_size=batch_size,
                ce_fn=ce_fn,
                tracker=tracker,
                output_path=os.path.join(output_path, "sbic_test/"),
                device=device
            )
            evaluate(
                is_testing=True,
                data_iter=ihc_test,
                test_name="ihc_test",
                model=model,
                batch_size=batch_size,
                ce_fn=ce_fn,
                tracker=tracker,
                output_path=os.path.join(output_path, "ihc_test/"),
                device=device
            )
            evaluate(
                is_testing=True,
                data_iter=dyna_test,
                test_name="dyna_test",
                model=model,
                batch_size=batch_size,
                ce_fn=ce_fn,
                tracker=tracker,
                output_path=os.path.join(output_path, "dyna_test/"),
                device=device
            )

  
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
src_path = os.path.abspath(os.path.join(notebook_dir, ".."))  # Move up to src/
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from modules import get_dataloader,get_dataloader_dynahate,get_dataloader_sbic,Metrics,HistoryTracker,SentenceTriplet,set_seed,prim_encoder_con,update_progress,load_progress,reset_progress,TrainingVisualizer,plot_confusion_matrix,plot_tsne
from easydict import EasyDict as edict
import torch 
from torch import nn
from tqdm import tqdm
import glob
import os
from itertools import product
from torch.optim import AdamW
from torch import nn
import torch
from transformers import get_linear_schedule_with_warmup
import json
import numpy as np


# In[2]:


batch_size=8


# In[3]:


train_iter,valid_iter,ihc_test= get_dataloader(train_batch_size=8,eval_batch_size=8,w_aug="imp",seed=0)
_,_,sbic_test= get_dataloader_sbic(train_batch_size=8,eval_batch_size=8,w_aug="imp",seed=0)
_,_,dyna_test= get_dataloader_dynahate(train_batch_size=8,eval_batch_size=8)


# In[ ]:


def test(test_loader,test_name, model, batch_size, ce_fn, tracker,output_path):
    model.cuda()
    model.eval()
    metrics = Metrics()
    total_loss = 0.0
    all_embeddings = []  # Store embeddings for t-SNE
    all_labels = []      # Store corresponding labels
    progress_bar = tqdm(test_loader, desc="Final Testing", unit="batch")
    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for batch in progress_bar:
            text = batch["post"].cuda()
            attn = batch["post_attn_mask"].cuda()
            label = torch.tensor(batch["label"]).long().cuda()
            og_hidden, og_feat = model.get_cls_features_ptrnsp(text, attn)  # Use og_feat for embeddings
            pred = model(og_hidden)  # Use og_hidden for classification
            loss = ce_fn(pred, label)
            total_loss += loss.item()
            # Store predictions and labels
            batch_preds = torch.argmax(pred, dim=1).detach()
            metrics.update(batch_preds, label.detach())
            # Store embeddings (og_feat) and labels for t-SNE
            all_embeddings.append(og_feat.cpu().numpy())  # Use og_feat instead of og_hidden
            all_labels.append(label.cpu().numpy())
            # Update progress bar
            progress_bar.set_postfix({"CE Loss": f"{loss.item():.4f}"})
    # Concatenate embeddings and labels
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Generate t-SNE plot
    plot_tsne(all_embeddings, all_labels, output_path)
    # Compute and save test results
    test_results = metrics.compute()
    test_results["loss"] = total_loss / len(test_loader)
    with open(os.path.join(output_path, f"{test_name}_result.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    # Plot confusion matrix
    plot_confusion_matrix(metrics, output_path)
    # Print results
    print("\n📊 Test Results:")
    for k, v in test_results.items():
        print(f"{k.capitalize()}: {v:.4f}")
    return test_results


# In[5]:


def validate(epoch, val_loader, model, batch_size, ce_fn, tracker, optimizer):
    model.cuda()
    model.eval()  # Evaluation mode
    metrics = Metrics()
    total_loss = 0.0
    progress_bar = tqdm(val_loader,
                        desc="😺 Validation in progress: ", 
                        unit="batch")    
    with torch.no_grad():
        for idx, batch in enumerate(progress_bar):
            text = batch["post"].cuda()
            attn = batch["post_attn_mask"].cuda()
            label = torch.tensor(batch["label"]).long().cuda()
            og_text= text
            og_attn= attn
            og_label=label
            og_hidden, _ = model.get_cls_features_ptrnsp(og_text, og_attn)
            pred = model(og_hidden)
            loss = ce_fn(pred, og_label)
            total_loss += loss.item()
            batch_preds = torch.argmax(pred, dim=1).detach()
            metrics.update(batch_preds, og_label.detach())
            progress_bar.set_postfix({"CE Loss": f"{loss.item():.4f}"})
    computed_metrics = metrics.compute()
    avg_loss = total_loss / len(val_loader)
    tracker.update_metrics(
        epoch=epoch,
        avg_loss=avg_loss,
        acc=computed_metrics["accuracy"],
        f1_m=computed_metrics["f1_macro"],
        f1_w=computed_metrics["f1_weighted"],
        precision=computed_metrics["precision"],
        recall=computed_metrics["recall"],
        is_validation=True  # Critical: mark as validation
    )
    # Save best model based on validation F1
    tracker.best_f1_score(
        epoch=epoch,
        current_f1=computed_metrics["f1_macro"],  
        model=model,
        optimizer=optimizer
    )
    tracker.save()
    return avg_loss,computed_metrics


# In[ ]:


def train(epoch, train_loader,val_loader, model, batch_size, lam,
          metric_fn, ce_fn, optimizer, lr_scheduler,
          tracker, metrics):
    model.cuda()
    model.train()
    metrics.reset()  # Reset metrics at the start of each epoch
    total_loss = 0
    progress_bar = tqdm(train_loader,
                        desc=f"Epoch {epoch} Progress", 
                        unit="batch")
    for idx, batch in enumerate(progress_bar):
        text = batch["post"].cuda()
        attn = batch["post_attn_mask"].cuda()
        label = torch.tensor(batch["label"]).long().cuda()
        if label.size(0) != batch_size * 2:
            continue
        og_text, ag_text = torch.split(text, [batch_size, batch_size], dim=0)
        og_attn, ag_attn = torch.split(attn, [batch_size, batch_size], dim=0)
        og_label, _ = torch.split(label, [batch_size, batch_size], dim=0)
        # Forward pass
        og_hidden, og_feat = model.get_cls_features_ptrnsp(og_text, og_attn)
        _, ag_feat = model.get_cls_features_ptrnsp(ag_text, ag_attn)
        pred = model(og_hidden)
        # Loss calculation
        ce = ce_fn(pred, og_label)  # Use original labels
        met = metric_fn(og_feat,ag_feat,og_label)
        loss = lam * ce + (1 - lam) * met

        
        # Update progress bar with current losses
        progress_bar.set_postfix({
            "mixed Loss": f"{loss.item():.4f}",
            "ce Loss": f"{ce.item():.4f}",
            "metric Loss": f"{met.item():.4f}",
        })
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        # Update batch-level losses and metrics
        tracker.update_loss(epoch, idx, loss.item(), ce.item(), met.item())
        total_loss += ce.item()
        
        # Collect predictions and labels for metrics
        batch_preds = torch.argmax(pred, dim=1).detach()
        metrics.update(batch_preds, og_label.detach())
    # Compute epoch-level metrics
    computed_train_metrics = metrics.compute()
    avg_train_loss = total_loss / len(train_loader)
    tracker.update_metrics(
        epoch=epoch,
        avg_loss=avg_train_loss,
        acc=computed_train_metrics["accuracy"],
        f1_m=computed_train_metrics["f1_macro"],
        f1_w=computed_train_metrics["f1_weighted"],
        precision=computed_train_metrics["precision"],
        recall=computed_train_metrics["recall"],
        is_validation=False
    )
    avg_loss_valid,computed_valid_metrics=validate(epoch, val_loader, model, batch_size, ce_fn, tracker, optimizer)
    print(f"Epoch {epoch} completed. \n"
        f"Training Loss: {avg_train_loss:.4f}, \n"
        f"Validation Loss: {avg_loss_valid:.4f}, \n"
        f"Training Accuracy: {computed_train_metrics['accuracy']:.2%}, \n" 
        f"Validation Accuracy: {computed_valid_metrics['accuracy']:.2%}, \n"
        f"Training F1 Score: {computed_train_metrics['f1_macro']:.2%}, \n"
        f"Validation F1 Score: {computed_valid_metrics['f1_macro']:.2%}\n")
    # Save model checkpoint and training history
    tracker.save_model(model, optimizer, epoch)
    
    tracker.save()
    return computed_valid_metrics["f1_macro"]


# In[ ]:


def run_pipeline(
    encoder_name="bert-base-uncased",
    learning_rate=5e-2,
    batch_size=8,
    lambda_weight=0.25,
    margin=0.3,
    num_epochs=6,
    output_base="./output",
    
    reducer = "mean",
    fallback = False,
    beta = 5
):
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_short_name = "bert" if "bert-base-uncased" in encoder_name else "hatebert"
    if reducer != "softmax" :
        output_path = f"{output_base}.semi_hard_reducers:{reducer}/fallback:{fallback}_{encoder_short_name}_lr:{learning_rate}_lam:{lambda_weight}_margin:{margin}_reducer:{reducer}/"
    else:
        output_path = f"{output_base}.semi_hard_reducers:{reducer}/beta:{beta}_fallback:{fallback}_{encoder_short_name}_lr:{learning_rate}_lam:{lambda_weight}_margin:{margin}_reducer:{reducer}/"
    os.makedirs(output_path, exist_ok=True)  # Add this line
    print(encoder_name)
    # Initialize model and optimizer
    model = prim_encoder_con(
        hidden_size=768,
        label_size=2,
        encoder_type=encoder_name
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Load tracker with existing history
    tracker = HistoryTracker.load(output_path)  
    if tracker.history["best"]["f1_macro"] is None:
        tracker.history["best"]["f1_macro"] = -1
    # Check for existing checkpoints
    start_epoch = 1
    latest_checkpoint = tracker.get_latest_checkpoint()
    if latest_checkpoint:
        try:
            start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1  
            model, optimizer, _ = tracker.load_model(latest_checkpoint, model, optimizer)
            print(f"✅ Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            start_epoch = 1
    # Initialize training components
    ce_fn = nn.CrossEntropyLoss()
    metric_fn = SentenceTriplet(margin=margin,reducers=reducer,use_fallback=fallback,beta=beta)
    num_training_steps = int(len(train_iter)*num_epochs)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    metrics = Metrics()
    # Training loop with automatic resume
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n🚀 Epoch {epoch}/{num_epochs}")
        # Train/validate for one epoch
        current_f1 = train(
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
        # Update best model if needed
        if tracker.best_f1_score(epoch, current_f1, model, optimizer):
            print(f"🏆 New best model at epoch {epoch} with F1: {current_f1:.4f}")
        tracker.save()
        print(f"💾 Saved checkpoint and metrics for epoch {epoch}")
    visualizer = TrainingVisualizer(tracker.history)
    visualizer.plot_metrics(output_path)
    print(tracker.history)
    if tracker.history["best"]["path"]:
        print("\n🔍 Testing with best model...")
        model, _, _ = tracker.load_model(tracker.history["best"]["path"], model)
        test_in_data = test(
            test_loader=ihc_test,
            test_name="in_data",
            model=model,
            batch_size=batch_size,
            ce_fn=ce_fn,
            tracker=tracker,
            output_path = os.path.join(output_path, "ihc_test/")

        )
        test_sbic = test(
            test_loader=sbic_test,
            test_name="sbic_test",
            model=model,
            batch_size=batch_size,
            ce_fn=ce_fn,
            tracker=tracker,
            output_path = os.path.join(output_path, "sbic_test/")
        )
        test_dyna = test(
            test_loader=dyna_test,
            test_name="dyna_test",
            model=model,
            batch_size=batch_size,
            ce_fn=ce_fn,
            tracker=tracker,
            output_path = os.path.join(output_path, "dyna_test/")
        )


# In[ ]:


encoders = ['bert-base-uncased']
learning_rates = [2e-5]
margins = [0.3]
betas= [15] # range bagus kira2 5 - 20 kalau 5 terlalu kecil dan tidak terlalu max, tapi kalau 20 akian seperti maks

reducers = ["softmax"]
# ###########
# best btw
# ##########
lambda_weights = [0.25]
fallback = True
best_val_loss = float('inf')
all_combinations = list(product(
    encoders,
    learning_rates,
    margins, lambda_weights,
    reducers,betas
))
progress_data = load_progress()
progress_data = progress_data or {}
if progress_data.get("total_combinations", 0) != len(all_combinations):
    print("Parameters changed or initial run. Resetting progress.")
    progress_data = {"last_completed_index": -1, 
                     "total_combinations": len(all_combinations)}
    update_progress(progress_data)
start_index = progress_data.get("last_completed_index", -1) + 1
for idx in range(start_index, len(all_combinations)):
    encoder_name, lr, margin, lam,reducer,beta = all_combinations[idx]
    print(f"Running combination {idx+1}/{len(all_combinations)}")
    # try:
    run_pipeline(
        encoder_name=encoder_name,
        batch_size=batch_size,
        learning_rate=lr,
        lambda_weight=lam,
        margin=margin,
        num_epochs=6,
        output_base="./output",
        reducer = reducer,
        fallback=fallback,
        beta=beta
    )
    # Update progress only if successful
    progress_data["last_completed_index"] = idx
    update_progress(progress_data)
    # except Exception as e:
    #     print(f"Error running combination {idx}: {e}")
    #     break  # Exit on error to avoid incorrect progress
if progress_data["last_completed_index"] >= len(all_combinations) - 1:
    reset_progress()


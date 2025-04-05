import os 
import numpy 
import json 
import tqdm 
from torch import nn 
import torch 
import re 
no_deprecation_warning=True
from modules import Metrics
from modules import plot_confusion_matrix,plot_tsne
from tqdm import tqdm
import numpy as np 
def evaluate(
    device,
    is_testing,  # True = testing mode, False = validation mode
    data_iter,
    model,
    ce_fn,
    tracker,
    # Validation-specific params (required when is_testing=False)
    optimizer=None,
    epoch=None,
    # Test-specific params (required when is_testing=True)
    test_name=None,
    output_path=None,
):
    model.to(device)
    model.eval()
    metrics = Metrics()
    total_loss = 0.0
    progress_desc = "📊 Testing" if is_testing else "🔄 Validating"
    # Initialize test-specific components
    if is_testing:
        all_embeddings = []
        all_labels = []
        os.makedirs(output_path, exist_ok=True)
    progress_bar = tqdm(data_iter, desc=progress_desc, unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
            # Prepare batch data
            text,attn_mask, labels = batch["post"].to(device), batch["post_attn_mask"].to(device),torch.tensor(batch["label"]).long().to(device)
            hidden_states, features = model.get_cls_features_ptrnsp(text, attn_mask)
            predictions = model(hidden_states)
            loss = ce_fn(predictions, labels)
            total_loss += loss.item()
            batch_preds = torch.argmax(predictions, dim=1).detach()
            metrics.update(batch_preds, labels.detach())
            # Store embeddings for visualization (test only)
            if is_testing:
                all_embeddings.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            # Update progress bar
            progress_bar.set_postfix({"CE Loss": f"{loss.item():.4f}"})
        # Final metrics calculation
        avg_loss = total_loss / len(data_iter)
        final_metrics = metrics.compute()

        if is_testing:
            # Test-specific processing
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            # Generate visualizations
            plot_tsne(all_embeddings, all_labels, output_path)
            plot_confusion_matrix(metrics, output_path)
            # Save results
            test_results = {k: v for k, v in final_metrics.items()}
            test_results["loss"] = avg_loss
            with open(os.path.join(output_path, f"{test_name}_results.json"), "w") as f:
                json.dump(test_results, f, indent=2)
            # Print formatted results
            print("\n📊 Final Test Results:")
            for metric, value in test_results.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            return test_results
        else:
            # Validation-specific processing
            tracker.update_metrics(
                epoch=epoch,
                avg_loss=avg_loss,
                acc=final_metrics["accuracy"],
                f1_m=final_metrics["f1_macro"],
                f1_w=final_metrics["f1_weighted"],
                precision=final_metrics["precision"],
                recall=final_metrics["recall"],
                is_validation=True
            )
            tracker.best_f1_score(
                epoch=epoch,
                current_f1=final_metrics["f1_macro"],
                model=model,
                optimizer=optimizer
            )
            tracker.save()
            return avg_loss, final_metrics

def train(device,
          method,
          epoch, 
          train_loader,
          val_loader, 
          model, 
          batch_size, 
          lam,
          metric_fn, 
          ce_fn, 
          optimizer, 
          lr_scheduler,
          tracker, 
          metrics):
    model.to(device)
    model.train()
    metrics.reset()  # Reset metrics at the start of each epoch
    total_loss = 0
    progress_bar = tqdm(train_loader,desc=f"Epoch {epoch} Progress", unit="batch")
    for idx, batch in enumerate(progress_bar):
        text = batch["post"].to(device)
        attn = batch["post_attn_mask"].to(device)
        label = torch.tensor(batch["label"]).long().to(device)
        if label.size(0) != batch_size * 2:
            continue
        og_text, ag_text = torch.split(text, [batch_size, batch_size], dim=0)
        og_attn, ag_attn = torch.split(attn, [batch_size, batch_size], dim=0)
        og_label, _ = torch.split(label, [batch_size, batch_size], dim=0)
        og_hidden, og_feat = model.get_cls_features_ptrnsp(og_text, og_attn)
        _, ag_feat = model.get_cls_features_ptrnsp(ag_text, ag_attn)
        pred = model(og_hidden)
        ce = ce_fn(pred, og_label) 
        if method =="semi-hard":
            metric_loss = metric_fn(og_feat,ag_feat,og_label)
        if method =="contrastive":
            sup_feat = torch.cat([og_feat, ag_feat])
            metric_loss=metric_fn(sup_feat)
        loss = lam * ce + (1 - lam) * metric_loss
        # Update progress bar with current losses
        progress_bar.set_postfix({
            "mixed Loss": f"{loss.item():.4f}",
            "ce Loss": f"{ce.item():.4f}",
            "metric Loss": f"{metric_loss.item():.4f}",
        })
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        # Update batch-level losses and metrics
        tracker.update_loss(epoch, idx, loss.item(), ce.item(), metric_loss.item())
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
    avg_loss_valid,computed_valid_metrics=evaluate(device=device,epoch=epoch, data_iter=val_loader, model=model, ce_fn=ce_fn, tracker=tracker, optimizer=optimizer,is_testing=False)
    print(f"Epoch {epoch} completed. \n"
        f"Training Loss: {avg_train_loss:.4f}, \n"
        f"Validation Loss: {avg_loss_valid:.4f}, \n"
        f"Training Accuracy: {computed_train_metrics['accuracy']:.2%}, \n" 
        f"Validation Accuracy: {computed_valid_metrics['accuracy']:.2%}, \n"
        f"Training F1 Score: {computed_train_metrics['f1_macro']:.2%}, \n"
        f"Validation F1 Score: {computed_valid_metrics['f1_macro']:.2%}\n")
    tracker.save_model(model, optimizer, epoch,lr_scheduler)
    tracker.save()
    return computed_valid_metrics["f1_macro"]



from torch import nn
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
class Metrics:
    def __init__(self):
        self.true_labels = []  # Consistent naming
        self.pred_labels = []  # Consistent naming
    def reset(self):
        self.true_labels.clear()
        self.pred_labels.clear()
    
    def update(self, batch_preds, batch_labels):
        """Store predictions and labels for later analysis"""
        self.pred_labels.extend(batch_preds.cpu().numpy())
        self.true_labels.extend(batch_labels.cpu().numpy())
    def compute(self):
        """Calculate metrics from stored predictions"""
        return {
            "accuracy": accuracy_score(self.true_labels, self.pred_labels),
            "precision": precision_score(self.true_labels, self.pred_labels, average='macro'),
            "recall": recall_score(self.true_labels, self.pred_labels, average='macro'),
            "f1_macro": f1_score(self.true_labels, self.pred_labels, average='macro'),
            "f1_weighted": f1_score(self.true_labels, self.pred_labels, average='weighted')
        }
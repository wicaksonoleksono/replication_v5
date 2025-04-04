import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from sklearn.manifold import TSNE
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class TrainingVisualizer:
    def __init__(self, history):
        self.history = history
        self.metrics = ['loss', 'acc', 'precision', 'recall', 'f1_macro']
        self.labels = {
            'loss': 'Cross-Entropy Loss',
            'acc': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_macro': 'F1 Score (Macro)'
        }

    def plot_metrics(self, output_path):
        """Generate separate plots for each metric with specified configurations"""
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare epoch-level data
        train_metrics, valid_metrics = self._prepare_epoch_data()
        
        # Generate plots for each metric
        for metric in self.metrics:
            plt.figure(figsize=(18, 12))
            
            if metric == 'loss':
                self._plot_loss(plt, train_metrics, valid_metrics)
            else:
                self._plot_standard_metric(plt, metric, train_metrics, valid_metrics)
            
            plot_path = os.path.join(output_path, f'{metric}_plot.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"📊 Saved {metric} plot to {plot_path}")

    def _prepare_epoch_data(self):
        """Extract epoch-level metrics for both train and validation"""
        train_metrics = []
        valid_metrics = []
        epochs = sorted(
            [int(k.split('_')[1]) for k in self.history['train'] if k.startswith('epoch_')],
            key=lambda x: x
        )

        for epoch in epochs:
            epoch_key = f'epoch_{epoch}'
            
            # Train metrics
            train_data = self.history['train'][epoch_key]
            # Calculate average training loss
            batches = train_data.get('losses', [])
            ce_losses = [b['ce_loss'] for b in batches]
            avg_train_loss = sum(ce_losses) / len(ce_losses) if ce_losses else None
            
            train_metrics.append({
                'epoch': epoch,
                'acc': train_data.get('acc'),
                'f1_macro': train_data.get('f1_macro'),
                'precision': train_data.get('precision'),
                'recall': train_data.get('recall'),
                'average_loss': avg_train_loss,
            })

            # Validation metrics
            valid_data = self.history['valid'][epoch_key]
            valid_metrics.append({
                'epoch': epoch,
                'acc': valid_data.get('acc'),
                'f1_macro': valid_data.get('f1_macro'),
                'precision': valid_data.get('precision'),
                'recall': valid_data.get('recall'),
                'average_loss': valid_data.get('average_loss'),
            })

        return pd.DataFrame(train_metrics), pd.DataFrame(valid_metrics)

    def _plot_loss(self, plt, train_df, valid_df):
        """Plot average training and validation CE loss per epoch"""
        # Plot training CE loss (epoch-level)
        plt.plot(train_df['epoch'], train_df['average_loss'], 
                 marker='o', linestyle='-', color='blue', 
                 linewidth=2, label='Train CE Loss')
        # Plot validation CE loss (epoch-level)
        plt.plot(valid_df['epoch'], valid_df['average_loss'], 
                 marker='o', linestyle='--', color='orange', 
                 linewidth=2, label='Validation CE Loss')

        plt.title(self.labels['loss'] + ' Evolution')
        plt.xlabel('Epoch')
        plt.ylabel(self.labels['loss'])
        plt.xticks(range(1, max(train_df['epoch']) + 1))
        plt.grid(True)
        plt.legend()

    def _plot_standard_metric(self, plt, metric, train_df, valid_df):
        """Handle standard epoch-level metrics"""
        plt.plot(train_df['epoch'], train_df[metric], 
                marker='o', linestyle='-', linewidth=2, 
                label=f'Train {self.labels[metric]}')
        
        plt.plot(valid_df['epoch'], valid_df[metric], 
                marker='o', linestyle='--', linewidth=2,
                label=f'Validation {self.labels[metric]}')

        plt.title(f'{self.labels[metric]} Evolution')
        plt.xlabel('Epoch')
        plt.ylabel(self.labels[metric])
        plt.xticks(range(1, max(valid_df['epoch'])+1))
        plt.grid(True)
        plt.legend()

def plot_confusion_matrix(metrics, output_path):
    """Plot using data from Metrics instance"""
    cm = confusion_matrix(metrics.true_labels, metrics.pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"📈 Saved confusion matrix to {cm_path}")


def plot_tsne(embeddings, labels, output_path):
    """Generate and save t-SNE plot for embeddings"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class Labels')
    plt.title('t-SNE Visualization of BERT Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    tsne_path = os.path.join(output_path, "tsne_plot.png")
    plt.savefig(tsne_path)
    plt.close()
    print(f"📊 Saved t-SNE plot to {tsne_path}")


from matplotlib import pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score, 
    RocCurveDisplay, 
    PrecisionRecallDisplay
)
import os
import numpy as np

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  output_dir: str, filename: str, title: str = 'ROC Curve') -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True binary labels
        y_prob: Target scores, can either be probability estimates or non-thresholded decision function
        output_dir: Directory to save the plot
        filename: Name of the output file (without extension)
        title: Plot title
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                 output_dir: str, filename: str, title: str = 'Precision-Recall Curve') -> None:
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_prob: Target scores, can either be probability estimates or non-thresholded decision function
        output_dir: Directory to save the plot
        filename: Name of the output file (without extension)
        title: Plot title
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Avg Precision = {avg_precision:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()
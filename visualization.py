import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def save_tsne_plot(embeddings, labels, epoch):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f't-SNE Plot - Epoch {epoch}')
    plt.savefig(f'Graphs/tsne_epoch_{epoch}.png')
    plt.close()

def save_confusion_matrix(true_labels, predicted_labels, epoch):
    # Check if labels are empty
    if not true_labels or not predicted_labels:
        print(f"Skipping confusion matrix for epoch {epoch}: true_labels or predicted_labels are empty.")
        return

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Check if confusion matrix is empty (zero-size array)
    if cm.size == 0:
        print(f"Skipping confusion matrix for epoch {epoch}: confusion matrix is empty.")
        return

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign', 'Normal'], 
                yticklabels=['Malignant', 'Benign', 'Normal'])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'Graphs/confusion_matrix_epoch_{epoch}.png')
    plt.close()

def save_roc_curve(y_true, y_pred_prob, epoch):
    # Check if y_true or y_pred_prob are empty
    if not y_true or not y_pred_prob:
        print(f"Skipping ROC curve for epoch {epoch}: y_true or y_pred_prob are empty.")
        return

    try:
        # Compute the ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch}')
        plt.legend(loc='lower right')
        plt.savefig(f'Graphs/roc_curve_epoch_{epoch}.png')
        plt.close()

    except ValueError as e:
        print(f"Error generating ROC curve for epoch {epoch}: {e}")


def save_augmented_samples(samples, augmentations, epoch):
    # Check if samples list is empty
    if not samples:
        print(f"Skipping augmented samples plot for epoch {epoch}: samples are empty.")
        return
    
    # Proceed only if samples are not empty
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    for ax, sample in zip(axes, samples):
        ax.imshow(sample)
        ax.axis('off')
    plt.suptitle(f'Augmented Samples - Epoch {epoch}')
    plt.savefig(f'Graphs/augmented_samples_epoch_{epoch}.png')
    plt.close()


def save_loss_vs_epoch(losses, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), losses, marker='o')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Graphs/loss_vs_epoch.png')
    plt.close()

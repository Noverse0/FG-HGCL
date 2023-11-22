import random
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_tsne(n, labels, dir="tsne_figure", file_name="tsne_plot.png", perplexity=30):
    """
    Plot t-SNE of the data and save to a file.
    
    Args:
        data: PyTorch Tensor or NumPy array of shape (num_samples, num_features)
        labels: PyTorch Tensor or NumPy array of shape (num_samples,) for color coding
        file_name: The name of the file to save the plot
        perplexity: t-SNE perplexity parameter
    """
    
    # Convert data to NumPy array if it's a tensor
    if isinstance(n, torch.Tensor):
        n = n.detach().cpu().numpy()
    
    # Convert labels to NumPy array if it's a tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=500, init='random', learning_rate=200.0).fit_transform(n)
    
    # Compute silhouette score
    s_score = silhouette_score(tsne, labels)
    
    # Get distinct colors for each label
    num_unique_labels = len(set(labels))
    cmap = plt.get_cmap('tab20', num_unique_labels)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap=cmap, s=30)
    plt.gca().set_facecolor('white')  # Set background color to white
    plt.grid(False)  # Remove grid

    # Adding the silhouette score to the plot
    plt.title(f"Silhouette score: {s_score:.3f}")

    # Save the plot
    plt.savefig(os.path.join(dir, file_name))
    plt.close()
    
import matplotlib.pyplot as plt
import numpy as np

def read_log_file(filename):
    with open(filename, 'r') as f:
        return np.array([float(line.strip()) for line in f])

def plot_metric(ax, data, title, color):
    epochs = np.arange(1, len(data) + 1)
    ax.plot(epochs, data, color=color)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.7)


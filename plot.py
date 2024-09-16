import matplotlib.pyplot as plt
import numpy as np
from utils import read_log_file, plot_metric

# Read log files
train_loss = read_log_file('logs/train_loss.txt')
train_perplex = read_log_file('logs/train_perplex.txt')
test_loss = read_log_file('logs/test_loss.txt')
test_perplex = read_log_file('logs/test_perplex.txt')

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Training and Evaluation Metrics', fontsize=16)

# Plot each metric
plot_metric(axs[0, 0], train_loss, 'Training Loss', 'blue')
plot_metric(axs[0, 1], train_perplex, 'Training Perplexity', 'green')
plot_metric(axs[1, 0], test_loss, 'Evaluation Loss', 'red')
plot_metric(axs[1, 1], test_perplex, 'Evaluation Perplexity', 'purple')

# Adjust layout and save
plt.tight_layout()
plt.show()
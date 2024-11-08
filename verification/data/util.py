import numpy as np
import torch


def get_loss_weights(device, train_label_statistic, power=1):
    per_class = np.array(train_label_statistic)
    weights_for_samples = 1.0 / np.array(np.power(per_class, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * len(train_label_statistic)
    normed_weights = torch.tensor(weights_for_samples, dtype=torch.float32, device=device)

    return normed_weights

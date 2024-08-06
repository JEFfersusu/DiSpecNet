import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np
import csv
import torch
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize
from ptflops import get_model_complexity_info
import torchvision.models as models
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import torch
import time
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize

configs = configs()

train_loader, val_loader = load_custom_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = configs.net.to(device)
checkpoint = torch.load(configs.test_weights)
net.load_state_dict(checkpoint)
net.eval()

test_metrics = {'loss': [], 'kappa': [], 'balanced_accuracy': [], 'precision_weighted': [], 'recall_weighted': [],
                'f1_weighted': [], 'auc_ovr': [], 'log_loss': [], 'duration': []}

def test():
    net.eval()
    test_loss, correct, total, start_time = 0, 0, 0, time.time()
    targets_all, predicted_all, probabilities_all = [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities_all.extend(probabilities.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            targets_all.extend(targets.cpu().numpy())
            predicted_all.extend(predicted.cpu().numpy())
    y_true_binarized = label_binarize(targets_all, classes=np.unique(targets_all))
    kappa = cohen_kappa_score(targets_all, predicted_all)
    balanced_acc = balanced_accuracy_score(targets_all, predicted_all)
    precision_weighted = precision_score(targets_all, predicted_all, average='weighted')
    recall_weighted = recall_score(targets_all, predicted_all, average='weighted')
    f1_weighted = f1_score(targets_all, predicted_all, average='weighted')
    auc_ovr = roc_auc_score(y_true_binarized, probabilities_all, average='weighted', multi_class='ovr')
    log_loss_val = log_loss(y_true_binarized, probabilities_all)
    test_metrics['loss'].append(test_loss / len(val_loader))
    test_metrics['kappa'].append(kappa)
    test_metrics['balanced_accuracy'].append(balanced_acc)
    test_metrics['precision_weighted'].append(precision_weighted)
    test_metrics['recall_weighted'].append(recall_weighted)
    test_metrics['f1_weighted'].append(f1_weighted)
    test_metrics['auc_ovr'].append(auc_ovr)
    test_metrics['log_loss'].append(log_loss_val)
    test_duration = time.time() - start_time
    test_metrics['duration'].append(test_duration)
    print(f'Test Loss: {test_loss / len(val_loader):.3f} | Accuracy: {100. * correct / total:.3f}% | '
          f'F1 Score (Weighted): {f1_weighted:.3f} | Kappa: {kappa:.3f} | '
          f'Balanced Accuracy: {balanced_acc:.3f} | Weighted Precision: {precision_weighted:.3f} | '
          f'Weighted Recall: {recall_weighted:.3f} | AUC OvR: {auc_ovr:.3f} | Log Loss: {log_loss_val:.3f} | '
          f'Duration: {test_duration:.3f} seconds')

test()

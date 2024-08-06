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
from data import *
from config import *

configs = configs()

train_loader, val_loader = load_custom_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = configs.net.to(device)
optimizer = configs.opt
scheduler = configs.sch
train_losses = []
val_losses = []
train_acc = []
val_acc = []
misclassified_samples_train = []
misclassified_samples_val = []
val_acc_best = 0
epoch_acc_best = 0
model_best = None
epoch_num = 100
criterion = configs.criterion
train_metrics = {'loss': [], 'kappa': [], 'balanced_accuracy': [], 'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': []}
val_metrics = {'loss': [], 'kappa': [], 'balanced_accuracy': [], 'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [], 'auc_ovr': [], 'log_loss': []}

def train(epoch):
    print('\nEpoch:', epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0
    targets_all, predicted_all = [], []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        targets_all.extend(targets.cpu().numpy())
        predicted_all.extend(predicted.cpu().numpy())
    kappa = cohen_kappa_score(targets_all, predicted_all)
    balanced_acc = balanced_accuracy_score(targets_all, predicted_all)
    precision_weighted = precision_score(targets_all, predicted_all, average='weighted')
    recall_weighted = recall_score(targets_all, predicted_all, average='weighted')
    f1_weighted = f1_score(targets_all, predicted_all, average='weighted')
    train_metrics['loss'].append(train_loss / len(train_loader))
    train_metrics['kappa'].append(kappa)
    train_metrics['balanced_accuracy'].append(balanced_acc)
    train_metrics['precision_weighted'].append(precision_weighted)
    train_metrics['recall_weighted'].append(recall_weighted)
    train_metrics['f1_weighted'].append(f1_weighted)
    print(f'Train Loss: {train_loss / len(train_loader):.3f} | Acc: {100. * correct / total:.3f}% | '
          f'Kappa: {kappa:.3f} | Balanced Acc: {balanced_acc:.3f} | '
          f'Weighted Precision: {precision_weighted:.3f} | Weighted Recall: {recall_weighted:.3f} | '
          f'Weighted F1: {f1_weighted:.3f}')
def val(epoch):
    global val_acc_best, epoch_acc_best, model_best
    net.eval()
    val_loss, correct, total = 0, 0, 0
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
    val_metrics['loss'].append(val_loss / len(val_loader))
    val_metrics['kappa'].append(kappa)
    val_metrics['balanced_accuracy'].append(balanced_acc)
    val_metrics['precision_weighted'].append(precision_weighted)
    val_metrics['recall_weighted'].append(recall_weighted)
    val_metrics['f1_weighted'].append(f1_weighted)
    val_metrics['auc_ovr'].append(auc_ovr)
    val_metrics['log_loss'].append(log_loss_val)
    if correct / total > val_acc_best:
        val_acc_best = correct / total
        epoch_acc_best = epoch
        model_best = net.state_dict()
    print(f'Val Loss: {val_loss / len(val_loader):.3f} | Acc: {100. * correct / total:.3f}% | '
          f'F1 Score (Weighted): {f1_weighted:.3f} | Kappa: {kappa:.3f} | '
          f'Balanced Acc: {balanced_acc:.3f} | Weighted Precision: {precision_weighted:.3f} | '
          f'Weighted Recall: {recall_weighted:.3f} | AUC OvR: {auc_ovr:.3f} | Log Loss: {log_loss_val:.3f}')
for epoch in range(0, epoch_num):
    train(epoch)
    val(epoch)
    scheduler.step()

def save_metrics_to_csv(metrics, file_name):
    epochs = range(1, len(next(iter(metrics.values()))) + 1)
    fieldnames = ['epoch'] + list(metrics.keys())
    with open(file_name, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in epochs:
            row = {'epoch': epoch}
            for key, values in metrics.items():
                row[key] = values[epoch - 1]
            writer.writerow(row)
save_metrics_to_csv(train_metrics, 'train.csv')
save_metrics_to_csv(val_metrics, 'val.csv')
save_path = 'net.pth'
torch.save(model_best, save_path)
print(f"最佳模型参数已保存至: {save_path}")
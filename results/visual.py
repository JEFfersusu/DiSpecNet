from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import *

configs = configs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = configs.net.to(device)
checkpoint = torch.load(configs.test_weights)  # Load the saved parameter file
net.load_state_dict(checkpoint)  # Loading model parameters
net.eval()

predictions = []
all_features = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = net(inputs.to(device))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        features = outputs.cpu().numpy()
        all_features.append(features)
        all_labels.extend(labels.numpy())
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
# Reduce dimensionality using t-SNE and visualize the results
features = np.concatenate(all_features, axis=0)
labels = np.array(all_labels)
tsne = TSNE(n_components=2, random_state=0)
embedded = tsne.fit_transform(features)
plt.figure(figsize=(10, 10))
for label in np.unique(labels):
    indices = np.where(labels == label)
    plt.scatter(embedded[indices, 0], embedded[indices, 1], label=label)
plt.legend()
plt.show()

# Calculate the confusion matrix and visualize the results
cm = confusion_matrix(labels, predictions, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
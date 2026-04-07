import sys
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt

sys.path.append("../../../../")

from elsanet.classifier import MultiVariableClassifier
from elsanet.metrics import Metrics
import json

# Setup hyperparams
# If 2 autoencoders are used per label, and latent space is 64, input is 128
# The number of classes depends on the COCO dataset categories
coco_file = "../../../data/coco640/train/_annotations.coco.json"
with open(coco_file, 'r') as f:
    coco_data = json.load(f)
n_classes = len(coco_data['categories']) + 1 # Include potentially a large enough count to cover the max ID
max_category_id = max([cat['id'] for cat in coco_data['categories']])
print(f"Número de classes detectado: {n_classes}, Max ID: {max_category_id}")

print("Carregando dataset codificado...")
data_instance = torch.load("ENCODED_DATA_COCO_SURFACE", weights_only=False)

inputs, targets = data_instance.data_train.dataset.tensors
input_feat_size = inputs.shape[-1]
print(f"Dimensão latente descoberta: {input_feat_size}")

model_hyperparameters = {
    "input_size": input_feat_size,          
    "n_classes": max_category_id + 1, 
    "epochs": 15,
    "learning_rate": 0.005,
    "loss_function": nn.NLLLoss(),
    "seed": 42,
}

classifier = MultiVariableClassifier(model_hyperparameters)

print("Iniciando treinamento do classificador...")
classifier.fit(data_instance)
print("Treinamento finalizado.")

plt.plot(classifier.loss_train, label="Loss train")
plt.plot(classifier.loss_val, label="Loss val")
plt.legend(loc="best")
plt.ylabel("Loss")
plt.xlabel("Épocas")
plt.savefig("Loss_Graph.png")
plt.clf()

with torch.no_grad():
    inputs, targets = data_instance.data_test.dataset.tensors
    classifier_out = classifier(inputs)
    classified = classifier_out.argmax(axis=-1).view(-1).to("cpu")

cm = confusion_matrix(targets, classified)

metrics = Metrics()
metrics.get_metrics(targets, classified, ["accuracy", "precision", "recall", "f1"])
print("Resultados finais (Métricas):")
print(metrics.metrics)

# Get class names for confusion matrix
class_names = ["" for _ in range(max_category_id + 1)]
for cat in coco_data['categories']:
    class_names[cat['id']] = cat['name']

# Filter only present classes for cleaner graph
present_classes = sorted(list(set(targets.numpy()) | set(classified.numpy())))
filtered_class_names = [class_names[i] if class_names[i] else str(i) for i in present_classes]

disp = ConfusionMatrixDisplay(confusion_matrix(targets, classified, labels=present_classes), display_labels=filtered_class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap="gray_r", ax=ax, xticks_rotation='vertical')
plt.xlabel("Label Previsto")
plt.ylabel("Label Real")
plt.tight_layout()
plt.savefig("ConfusionMatrix_COCO.png")
print("Matriz de confusão salva como ConfusionMatrix_COCO.png e relatórios gerados.")

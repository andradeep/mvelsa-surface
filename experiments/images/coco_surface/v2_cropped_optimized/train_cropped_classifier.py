import sys
import os
sys.path.append("../../../../")

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

from data.data_preparation import DataPreparation
from elsanet.classifier import MultiVariableClassifier

# Load previously encoded variables
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
data_instance = torch.load("ENCODED_DATA_CROPPED_SURFACE", weights_only=False)

# Descobrir dispositivo (CUDA ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensão latente descoberta
inputs, targets = data_instance.data_train.dataset.tensors
input_feat_size = inputs.shape[-1]
print(f"Dimensão latente descoberta: {input_feat_size}")

# OTIMIZAÇÃO DE TESE 4: Class Weights (Suavizados para evitar explosão)
y_train = targets.numpy()
classes_unique = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)

# CAP nos pesos: Se uma classe for rara demais (ex: SHIP), não deixamos o peso passar de 3.0
# Isso evita que o modelo "chute" SHIP em tudo para baixar o loss.
class_weights = np.clip(class_weights, 0.5, 3.0) 

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Pesos de classe suavizados (Capped at 3.0): {class_weights_tensor}")

model_parameters = {
    "input_size": input_feat_size,
    "n_classes": len(data_instance.labels_list),
    "learning_rate": 0.001,
    "epochs": 100,
    "loss_function": nn.NLLLoss(weight=class_weights_tensor), # Pesos movidos para o device correto
    "seed": 42,
}

print(f"Número de classes identificadas no dataset latente: {model_parameters['n_classes']}")

classifier = MultiVariableClassifier(model_parameters)

print("INICIANDO TREINAMENTO DO CLASSIFICADOR (CROPPED DATASET)")
classifier.fit(data_instance)

# Plots
classifier_loss_fig = plt.figure()
plt.plot(classifier.loss_train, label="Train Loss")
plt.plot(classifier.loss_val, label="Validation Loss", linestyle="--")
plt.title(f"Classifier Loss (Cropped) - Time: {classifier.elapsed_time:.2f}s")
plt.xlabel("Steps")
plt.ylabel("NLL Loss")
plt.legend()
classifier_loss_fig.savefig("Cropped_Loss_Graph.png")
print("Saved Cropped_Loss_Graph.png")

# Predict over the COMPLETE test set
test_loader = data_instance.data_test
all_predictions = []
all_targets = []
all_probs = []

classifier.eval() # Set to evaluation mode
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        classifier_out = classifier(inputs)
        # Reshape output if needed (batch, classes)
        classified = classifier_out.view(classifier_out.shape[0], classifier_out.shape[-1])
        
        # Get probabilities for mAP/ROC (LogSoftmax -> Softmax)
        probs = torch.exp(classified)
        
        preds = torch.argmax(classified, dim=1).cpu().numpy()
        
        all_predictions.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

predictions = np.array(all_predictions)
targets = np.array(all_targets)
probabilities = np.array(all_probs)

acc = accuracy_score(targets, predictions)
recall = recall_score(targets, predictions, average="macro")
precision = precision_score(targets, predictions, average="macro")

print("\n--- PERFORMANCE NO CONJUNTO DE TESTE ---")
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")

# --- OTIMIZAÇÃO DE TESE: MÉTRICAS AVANÇADAS (mAP, PR, ROC) ---
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarizar labels para multiclasse (necessário para mAP e curvas por classe)
n_classes = len(data_instance.labels_list)
targets_bin = label_binarize(targets, classes=range(n_classes))

# Matrix Labels
classes_names = []
base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
for label in data_instance.labels_list:
    classes_names.append(base_classes.get(label, str(label)))

# 1. Calcular mAP (Mean Average Precision)
# MVELSA aqui atua como classificador; o mAP é a média das APs de cada classe.
aps = []
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    ap = average_precision_score(targets_bin[:, i], probabilities[:, i])
    aps.append(ap)
    
    precision_vals, recall_vals, _ = precision_recall_curve(targets_bin[:, i], probabilities[:, i])
    plt.plot(recall_vals, precision_vals, label=f'{classes_names[i]} (AP={ap:.2f})')

mAP = np.mean(aps)
print(f"Mean Average Precision (mAP): {mAP:.4f}")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (mAP={mAP:.4f})')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("Cropped_PR_Curve.png")
print("Saved Cropped_PR_Curve.png")

# 2. ROC Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(targets_bin[:, i], probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes_names[i]} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Multiclass)')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("Cropped_ROC_Curve.png")
print("Saved Cropped_ROC_Curve.png")

# 3. Matriz de Confusão NORMALIZADA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print(f"\nMatrix Labels (in order): {classes_names}")

cm = confusion_matrix(targets, predictions, labels=range(n_classes), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_names)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap="gray_r", ax=ax, xticks_rotation=45, values_format='.2f')

# OTIMIZAÇÃO DE TESE: Adicionar métricas globais no título do gráfico
plt.title(f"Confusion Matrix Normalized (All Marine Classes)\n"
          f"Accuracy: {acc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}", 
          fontsize=14, pad=20)

plt.tight_layout()
fig.savefig("Cropped_ConfusionMatrix.png")
print("\nMatriz de Confusão NORMALIZADA (com métricas) salva como Cropped_ConfusionMatrix.png")

# SALVAR MODELO PARA V3 (DETECÇÃO) - OTIMIZADO (Apenas pesos)
checkpoint = {
    'state_dict': classifier.state_dict(),
    'model_parameters': model_parameters,
    'labels_list': data_instance.labels_list
}
torch.save(checkpoint, "MVELSA_CLASSIFIER.pth")
print("Modelo de classificação (pesos apenas) salvo como MVELSA_CLASSIFIER.pth para uso na v3.")

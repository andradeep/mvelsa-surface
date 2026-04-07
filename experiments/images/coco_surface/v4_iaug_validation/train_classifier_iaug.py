"""
MVELSA V4 — Treinamento do Classificador MLP (IAUG)
====================================================
Treina o classificador multi-variável nos vetores latentes gerados pelo ELSA_MODEL_IAUG.
"""
import sys
import os
sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")  # necessário para desserializar ENCODED_DATA_IAUG

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

from elsanet.classifier import MultiVariableClassifier

# Carregar representações latentes
print("Carregando ENCODED_DATA_IAUG...")
data_instance = torch.load("ENCODED_DATA_IAUG", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

inputs, targets = data_instance.data_train.dataset.tensors
input_feat_size = inputs.shape[-1]
print(f"Dimensão latente: {input_feat_size}")

# Class weights (balanceados e capados)
y_train = targets.numpy()
classes_unique = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
class_weights = np.clip(class_weights, 0.5, 3.0)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
print(f"Classes: {[base_classes.get(int(c), c) for c in classes_unique]}")
print(f"Pesos (capped 0.5-3.0): {class_weights_tensor.tolist()}")

model_parameters = {
    "input_size":    input_feat_size,
    "n_classes":     len(data_instance.labels_list),
    "learning_rate": 0.001,
    "epochs":        100,
    "loss_function": nn.NLLLoss(weight=class_weights_tensor),
    "seed":          42,
}

classifier = MultiVariableClassifier(model_parameters)

print("\nINICIANDO TREINAMENTO DO CLASSIFICADOR V4 (IAUG)")
classifier.fit(data_instance)

# --- Gráfico de loss ---
fig = plt.figure()
plt.plot(classifier.loss_train, label="Train Loss")
plt.plot(classifier.loss_val, label="Val Loss", linestyle="--")
plt.title(f"Classifier Loss V4 IAUG - {classifier.elapsed_time:.2f}s")
plt.xlabel("Steps")
plt.ylabel("NLL Loss")
plt.legend()
fig.savefig("IAUG_Loss_Graph.png")
print("Salvo IAUG_Loss_Graph.png")

# --- Avaliação no test (fullHD633 val) ---
all_predictions, all_targets = [], []
classifier.eval()
with torch.no_grad():
    for batch in data_instance.data_test:
        inp, tgt = batch
        out  = classifier(inp).view(inp.shape[0], -1)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_predictions.extend(preds)
        all_targets.extend(tgt.cpu().numpy())

predictions = np.array(all_predictions)
tgts        = np.array(all_targets)

acc = accuracy_score(tgts, predictions)
print(f"\n--- PERFORMANCE NO CONJUNTO DE VALIDAÇÃO (fullHD633) ---")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Recall:    {recall_score(tgts, predictions, average='macro', zero_division=0):.4f}")
print(f"  Precision: {precision_score(tgts, predictions, average='macro', zero_division=0):.4f}")

# Salvar modelo
checkpoint = {
    'state_dict':       classifier.state_dict(),
    'model_parameters': model_parameters,
    'labels_list':      data_instance.labels_list,
}
torch.save(checkpoint, "MVELSA_CLASSIFIER_IAUG.pth")
print("\nModelo salvo como MVELSA_CLASSIFIER_IAUG.pth")
print("Próximo passo: python calibrate_experts_iaug.py")

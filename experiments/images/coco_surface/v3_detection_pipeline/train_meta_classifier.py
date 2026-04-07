"""
MVELSA-REP: Meta-Classificador (Random Forest)
===============================================
Treina um Random Forest nos perfis REP de 50D (10 features × 5 especialistas).
Gera feature importance plot para a tese.
"""
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# --- 1. CARREGAR PERFIS ---
print("Carregando perfis REP...")
train_data = torch.load('rep_profiles_train.pt', weights_only=False)
val_data   = torch.load('rep_profiles_val.pt', weights_only=False)

X_train = train_data['profiles'].numpy()
y_train = train_data['labels'].numpy()
X_val   = val_data['profiles'].numpy()
y_val   = val_data['labels'].numpy()

labels_list    = train_data['labels_list']
n_experts      = train_data['n_experts']
feat_per_exp   = train_data['features_per_expert']
global_feats   = train_data.get('global_features', 0)

print(f"Train: {X_train.shape} | Val: {X_val.shape}")
print(f"Experts: {n_experts} | Features/expert: {feat_per_exp} | Global: {global_feats}")

# Nomes das classes
base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
EVAL_IDS   = sorted(base_classes.keys())
EVAL_NAMES = [base_classes[k] for k in EVAL_IDS]

# --- 2. NOMES DAS FEATURES (para interpretabilidade) ---
feature_names = []
for idx, label in enumerate(labels_list):
    cls = base_classes.get(label, f"ID_{label}")
    for feat in ['MSE_global', 'MSE_R', 'MSE_G', 'MSE_B',
                 'MSE_TL', 'MSE_TR', 'MSE_BL', 'MSE_BR',
                 'SSIM', 'Latent_Dist']:
        feature_names.append(f"{cls}_{feat}")
if global_feats > 0:
    feature_names.append('cy_norm')

print(f"Total features: {len(feature_names)}")

# --- 3. TREINAR RANDOM FOREST ---
print("\nTreinando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Cross-validation no train
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- 4. AVALIAR NO VALIDATION ---
y_pred = rf.predict(X_val)

# Filtrar apenas EVAL_CLASSES
mask = np.isin(y_val, EVAL_IDS)
y_val_f = y_val[mask]
y_pred_f = y_pred[mask]

print(f"\n{'='*50}")
print("✅ MVELSA-REP — Random Forest no Perfil de Reconstrução (50D)")
print('='*50)
print(classification_report(y_val_f, y_pred_f, labels=EVAL_IDS, target_names=EVAL_NAMES, zero_division=0))
cm = confusion_matrix(y_val_f, y_pred_f, labels=EVAL_IDS)
print("Matriz de Confusão (BOAT | BUOY | LAND | SHIP | SKY):")
print(cm)
acc = accuracy_score(y_val_f, y_pred_f)
print(f"\nAcurácia MVELSA-REP: {acc:.4f}")
print(f"Baseline aleatório:  {1/len(EVAL_IDS):.4f}")

# --- 5. FEATURE IMPORTANCE (para a tese) ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 15 features
top_n = min(15, len(feature_names))
plt.figure(figsize=(12, 6))
plt.title("MVELSA-REP: Feature Importance (Top 15)")
plt.bar(range(top_n), importances[indices[:top_n]], align='center')
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
plt.ylabel("Importância")
plt.tight_layout()
plt.savefig("REP_Feature_Importance.png", dpi=150)
print("\n📊 Feature Importance salva em: REP_Feature_Importance.png")

# Print top features
print("\nTop 10 Features Mais Importantes:")
for i in range(min(10, len(feature_names))):
    print(f"  {i+1}. {feature_names[indices[i]]:25s} = {importances[indices[i]]:.4f}")

# --- 6. SALVAR MODELO ---
with open('rep_meta_classifier.pkl', 'wb') as f:
    pickle.dump({
        'model': rf,
        'feature_names': feature_names,
        'labels_list': labels_list,
        'eval_ids': EVAL_IDS,
        'eval_names': EVAL_NAMES,
        'accuracy': acc,
        'cv_scores': cv_scores.tolist(),
    }, f)
print("\n✅ Meta-classificador salvo em: rep_meta_classifier.pkl")

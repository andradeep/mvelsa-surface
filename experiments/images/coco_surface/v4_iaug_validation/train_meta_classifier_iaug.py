"""
MVELSA V4 — Meta-Classificador Random Forest (IAUG)
====================================================
Treina RF nos perfis REP extraídos do IAUG train.
Valida no fullHD633 val (199 imagens — mesma base da V3: 77.89%).

Hipótese: treino com dados augmentativamente gerados melhora acurácia.
"""
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

print("Carregando perfis REP IAUG...")
train_data = torch.load('rep_profiles_train_iaug.pt', weights_only=False)
val_data   = torch.load('rep_profiles_val_iaug.pt',   weights_only=False)

X_train = train_data['profiles'].numpy()
y_train = train_data['labels'].numpy()
X_val   = val_data['profiles'].numpy()
y_val   = val_data['labels'].numpy()

labels_list  = train_data['labels_list']
n_experts    = train_data['n_experts']
feat_per_exp = train_data['features_per_expert']
global_feats = train_data.get('global_features', 0)

print(f"Train (IAUG): {X_train.shape} | Val (fullHD633): {X_val.shape}")
print(f"Experts: {n_experts} | Features/expert: {feat_per_exp} | Global: {global_feats}")

base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
EVAL_IDS     = sorted(base_classes.keys())
EVAL_NAMES   = [base_classes[k] for k in EVAL_IDS]

# Nomes das features (para interpretabilidade)
feature_names = []
for label in labels_list:
    cls = base_classes.get(label, f"ID_{label}")
    for feat in ['MSE_global', 'MSE_R', 'MSE_G', 'MSE_B',
                 'MSE_TL', 'MSE_TR', 'MSE_BL', 'MSE_BR',
                 'SSIM', 'Latent_Dist']:
        feature_names.append(f"{cls}_{feat}")
if global_feats > 0:
    feature_names.append('cy_norm')

print(f"Total features: {len(feature_names)}")

# --- Treinar Random Forest ---
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

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy (5-fold, IAUG train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Avaliar no fullHD633 val ---
y_pred = rf.predict(X_val)
mask   = np.isin(y_val, EVAL_IDS)
y_val_f  = y_val[mask]
y_pred_f = y_pred[mask]

print(f"\n{'='*60}")
print("✅ MVELSA-REP V4 — Random Forest (IAUG train → fullHD633 val)")
print('='*60)
print(classification_report(y_val_f, y_pred_f, labels=EVAL_IDS, target_names=EVAL_NAMES, zero_division=0))
cm  = confusion_matrix(y_val_f, y_pred_f, labels=EVAL_IDS)
print("Matriz de Confusão (BOAT | BUOY | LAND | SHIP | SKY):")
print(cm)
acc = accuracy_score(y_val_f, y_pred_f)
print(f"\nAcurácia MVELSA-REP V4: {acc:.4f}")
print(f"Baseline aleatório:      {1/len(EVAL_IDS):.4f}")
print(f"\n[Referência V3: 0.7789]")
delta = acc - 0.7789
print(f"Delta V4 vs V3: {delta:+.4f} ({'✅ melhora' if delta > 0 else '❌ piora'})")

# --- Feature Importance ---
importances = rf.feature_importances_
indices     = np.argsort(importances)[::-1]
top_n       = min(15, len(feature_names))

plt.figure(figsize=(12, 6))
plt.title("MVELSA-REP V4 (IAUG): Feature Importance (Top 15)")
plt.bar(range(top_n), importances[indices[:top_n]], align='center')
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
plt.ylabel("Importância")
plt.tight_layout()
plt.savefig("IAUG_REP_Feature_Importance.png", dpi=150)
print("\n📊 Feature Importance salva em: IAUG_REP_Feature_Importance.png")

print("\nTop 10 Features Mais Importantes:")
for i in range(min(10, len(feature_names))):
    print(f"  {i+1}. {feature_names[indices[i]]:25s} = {importances[indices[i]]:.4f}")

# --- Salvar modelo ---
with open('rep_meta_classifier_iaug.pkl', 'wb') as f:
    pickle.dump({
        'model':        rf,
        'feature_names': feature_names,
        'labels_list':  labels_list,
        'eval_ids':     EVAL_IDS,
        'eval_names':   EVAL_NAMES,
        'accuracy':     acc,
        'cv_scores':    cv_scores.tolist(),
        'v3_reference': 0.7789,
    }, f)
print("\n✅ Meta-classificador salvo em: rep_meta_classifier_iaug.pkl")
print("Próximo passo: python validate_iaug_blind.py")

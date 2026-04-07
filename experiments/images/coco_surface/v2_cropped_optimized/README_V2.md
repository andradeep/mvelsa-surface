# MVELSA V2: Cropped & Optimized (Thesis Level)

Esta é a versão principal de desenvolvimento para a tese, focada em objetos de interesse (recortes) e métricas de alta precisão.

## Resultados Atuais (Sujeitos a Re-treinamento)
- **Matriz de Confusão:** [Cropped_ConfusionMatrix.png](./Cropped_ConfusionMatrix.png)
- **Curva Precision-Recall:** [Cropped_PR_Curve.png](./Cropped_PR_Curve.png)
- **Curva ROC:** [Cropped_ROC_Curve.png](./Cropped_ROC_Curve.png)
- **Desempenho:** ~96% (Auditado com Scene-Level Split).

## Passo a Passo de Funcionamento

### 1. Preparação do Dataset (`create_cropped_dataset.py`)
Gera recortes das 6 classes alvo e realiza o **Scene-Level Split** (70/15/15), garantindo que imagens de uma mesma cena não se misturem entre treino e teste.

### 2. Treinamento Deep ELSA (`train_cropped_mvelsa.py`)
Treina o Autoencoder profundo em RGB (64x64x3) para extração de features latentes.

### 3. Codificação (`gen_cropped_encoded.py`)
Extrai as variáveis latentes do dataset completo utilizando o modelo ELSA treinado.

### 4. Classificador Final (`train_cropped_classifier.py`)
Treina a rede de classificação multiclasse com pesos balanceados e gera os gráficos de tese.

---
*Importante: Sempre execute os scripts de dentro desta pasta para garantir que os caminhos relativos funcionem.*

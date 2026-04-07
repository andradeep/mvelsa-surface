# MVELSA V3: Trial de Objetos Avulsos

Esta versão é focada na validação rápida do classificador MVELSA utilizando imagens de objetos únicos (já isolados).

## Como Funciona

O script `predict_trial.py` realiza a inferência em lote:
1.  **Pasta de Entrada:** Ele busca imagens (`.jpg`, `.png`, etc.) dentro do diretório `trial/`.
2.  **Inferência:** Cada imagem é processada pela rede ELSA e pelo Classificador final da **V2**.
3.  **Resultado:** Gera um arquivo **`results.txt`** listando o nome de cada arquivo seguido pelo objeto identificado e o nível de confiança.

## Como Usar

### 1. Preparação
- Certifique-se de que o treinamento da **V2** foi concluído e os modelos estão na pasta ao lado.
- Coloque suas imagens de teste dentro da pasta `v3_detection_pipeline/trial/`.

### 2. Execução
```bash
python3 predict_trial.py
```

### 3. Verificação
Abra o arquivo `results.txt` gerado na raiz desta pasta para ver as predições.

---
*Esta fase permite validar de forma limpa a capacidade de generalização do modelo em dados externos (Trial) antes de avançar para detectores automáticos.*

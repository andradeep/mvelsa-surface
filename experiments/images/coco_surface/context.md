# Contexto do Projeto: Otimização MVELSA para Tese de Doutorado

**Última Atualização:** 11 de Março de 2026 - 06:05 AM

Este arquivo serve como um ponto de controle (checkpoint) de todo o progresso, arquitetura atual e decisões metodológicas adotadas para resolver os problemas de performance do modelo MVELSA no dataset COCO adaptado para ambiente marítimo.

---

## 1. Dicionário de Arquivos e Dependências
Para manter o controle sobre a complexidade do projeto, aqui está a função de cada arquivo na pasta `coco_surface`:

### Scripts de Execução (Pipeline)
*   **`create_cropped_dataset.py`**: [PREPARADOR] Lê as anotações JSON do COCO original e gera milhares de imagens PNG individuais para cada Bounding Box (recortes).
*   **`cropped_data_generator.py`**: [LOADER] Classe de Dataset do PyTorch que carrega os recortes em RGB e aplica Data Augmentation em tempo real.
*   **`train_cropped_mvelsa.py`**: [TREINADOR AE] O coração do MVELSA. Treina os autoencoders profundos para reconstruir as classes foco (Boat, Buoy, Ship).
*   **`gen_cropped_encoded.py`**: [EXTRATOR] Usa os autoencoders treinados para ler o dataset e salvar um arquivo binário com os vetores latentes.
*   **`train_cropped_classifier.py`**: [CLASSIFICADOR] Treina a rede final que decide se o vetor é um Barco, Bóia ou Navio, usando pesos para compensar o desbalanceamento.

### Arquivos de Dados e Metadados
*   **`COCO_CROPPED_METADATA.csv`**: Índice contendo o caminho de cada imagem recortada e o seu respectivo ID de classe.
*   **`ENCODED_DATA_CROPPED_SURFACE`**: Arquivo binário (Torch save) que armazena os vetores extraídos pelo ELSA. É o "cérebro" latente que o classificador lê.
*   **`context.md`**: Este arquivo de checkpoint e documentação de suporte.
*   **`README.md`**: Documentação de alto nível para humanos.

---

## 2. A Solução Aplicada (Otimizações de Doutorado Nível 4)

Implementamos um pipeline avançado focado no subconjunto ultra-relevante:
*   **Classes Foco:** `1 (BOAT)`, `3 (BUOY)`, `10 (SHIP)`

### Otimização 1: Preservação de Atributos RGB
As imagens são lidas em 3 canais. Bóias e barcos agora possuem cores que os distinguem do mar azul/cinza.

### Otimização 2: Data Augmentation Geométrica
Injeção de `RandomHorizontalFlip` e `ColorJitter` para simular um dataset maior e evitar que a rede decore fotos específicas (combate ao overfitting).

### Otimização 3: Rede ELSA Profunda
Arquitetura de transição suave: `[12288 -> 1024 -> 256 -> 128]`. Maior capacidade de abstração para os 3 canais de cor.

### Otimização 4: Punição Categórica (Class Weights)
Uso de `compute_class_weight` no Classificador. Errar um "SHIP" custa matematicamente muito caro para a rede, forçando-a a aprender a classe minoritária.

---

## 3. Estado de Execução Atual do Computador
(Se o sistema reiniciar, retome a partir deste ponto)

1.  **MVELSA Autoencoder:** Atualmente treinando (Ponto de Label 3: BUOY - ~70%).
2.  **Filtro Ativo:** BOAT (1), BUOY (3), SHIP (10). Outros labels são ignorados.
3.  **Ambiente:** `vision-env` ativado.
4.  **Hardware:** Rodando em CPU/DirectML (Compatibilidade Linux via CPU no momento).

# MVELSA Experiments: COCO Surface

Este diretório contém os experimentos de percepção marítima utilizando o modelo MVELSA no dataset COCO adaptado.

## Organização do Repositório

### [v1_full_image](./v1_full_image)
Primeira fase do projeto. Avaliação do MVELSA utilizando imagens completas (640x640). Focado no baseline inicial.
- **Modelos:** ELSA_MODEL_COCO_SURFACE
- **Destaque:** Baseline de 3 classes original.

### [v2_cropped_optimized](./v2_cropped_optimized)
Fase atual de **Doutorado/Tese**. Utiliza recortes individuais (Objects of Interest), split estratificado por imagem para evitar leakage e métricas avançadas (mAP, PR, ROC).
- **Modelos:** ELSA_MODEL_CROPPED_SURFACE
- **Destaque:** Multiclasse (6 classes) com 96% de performance potencial.

## Documentação e Utilitários
- [auxiliary.md](./auxiliary.md): Descrição de scripts de suporte e utilitários.
- [mvelsa_comparative_analysis.md](./mvelsa_comparative_analysis.md): Comparativo técnico entre v1 e v2.
- [mvelsa_full_comparative_table.md](./mvelsa_full_comparative_table.md): Tabela detalhada de métricas.

---
*Para rodar os modelos atuais, utilize os scripts dentro da pasta `v2_cropped_optimized`.*

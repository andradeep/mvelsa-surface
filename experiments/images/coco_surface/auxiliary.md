# Scripts Auxiliares e Utilitários

Este documento descreve os scripts de suporte utilizados para análise de dados e verificação de integridade no projeto MVELSA.

## Análise e Debug

### `check_labels.py`
Utilitário para verificar a consistência dos arquivos `labels.csv` no dataset cortado. Ele valida se as IDs das classes correspondem aos nomes e se as imagens referenciadas existem no disco.

### `count_all.py`
Script rápido para contagem total de objetos por classe em todo o repositório. Útil para gerar as tabelas de composição do dataset.

### `context.md`
Arquivo de notas rápidas sobre o ambiente de execução, versões de bibliotecas e caminhos de sistema.

## Documentação de Resultados

### `mvelsa_comparative_analysis.md`
Análise detalhada comparando o desempenho entre o modelo original (Imagens Inteiras) e o modelo otimizado de Doutorado (Cropped). Contém as justificativas para a mudança de estratégia.

### `mvelsa_full_comparative_table.md`
Tabela técnica consolidada com métricas de Precision, Recall e mAP por classe em cada versão do modelo.

---
*Estes scripts não fazem parte do pipeline principal de treinamento, mas são essenciais para a validade científica da tese.*

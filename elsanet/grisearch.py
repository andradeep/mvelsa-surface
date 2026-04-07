# Tamanho da janela de entrada em quantidade de amostras
# I = { 7, 14, 21, 28 } (1, 2, 3 e 4 semanas)

# Quantidade de autoencoders dentro da rede (MV)ELSA
# A = { 1, 2, 3, 4 }.
# A = 1 não configura uma rede ELSA.
# Avaliar se seria bom inserir o modelo equivalente ao A = 2

# Tamanho do espaço latente de cada um dos autoencoders internos do ELSA
# S = { 10, 15, 20, 25}.

# Quantidade de camadas internas de cada um dos autoencoders do ELSA
# L = { 1, 2 }.

# Tamanho de cada camada interna em cada um  dos autoencoders do ELSA
# Q = { 10, 20, 30 }.

# Valores para seed (trail)
# P = { 1, 2, 3, ..., 20 } (a depender da demora, talvez só 10)

# Isso aqui está relacionado ao preditor.
# Tamanho da janela de saída (predição) em quantidade de amostras
# I = { 7, 14, 21, 28 } (1, 2, 3 e 4 semanas)

# O que não for citado aqui, terá valores fixos.

valores = {
    "input_window_size": [7, 14, 21, 28],
    "forecast_window_size": [7, 14, 21, 28],
    "autoencoders_quantity": [2, 3],
    "latent_space_dimension": [10, 15, 30, 40],
    "internal_layers_quantity": [1, 2],
    "internal_layers_dimension": [10, 20],
    "seed": range(10),
}

# Talvez tenham muitas variações, e demore demais o treinamento.

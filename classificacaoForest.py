import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import glob

# Função para calcular a homogeneidade
def calcular_homogeneidade(imagem):
    glcm = graycomatrix(imagem, distances=[1], angles=[0], symmetric=True, normed=True)
    homogeneidade = graycoprops(glcm, 'homogeneity')[0, 0]
    return homogeneidade

# Função para converter imagem de BGR para CMYK
def bgr_to_cmyk(imagem):
    b, g, r = cv2.split(imagem)
    c = 1 - r / 255.0
    m = 1 - g / 255.0
    y = 1 - b / 255.0
    k = np.minimum(c, np.minimum(m, y))
    c = (c - k) / (1 - k + 1e-10)
    m = (m - k) / (1 - k + 1e-10)
    y = (y - k) / (1 - k + 1e-10)
    cmyk = cv2.merge((c, m, y, k))
    return (cmyk * 255).astype(np.uint8)

# Função para calcular descritores de cor
def calcular_descritores_de_cor(imagem):
    canais = cv2.split(imagem)
    descritores = []
    for canal in canais:
        media = np.mean(canal)
        desvio_padrao = np.std(canal)
        descritores.extend([media, desvio_padrao])
    return descritores

# Função para extrair características de uma imagem
def extrair_caracteristicas(imagem):
    imagem_cmyk = bgr_to_cmyk(imagem)
    _, _, y, k = cv2.split(imagem_cmyk)
    
    # Segmentação usando thresholding
    _, imagem_binaria = cv2.threshold(k, 130, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    imagem_binaria = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)
    
    # Calcular homogeneidade
    homogeneidade = calcular_homogeneidade(imagem_binaria)
    
    # Calcular descritores de cor
    descritores_de_cor = calcular_descritores_de_cor(imagem)
    
    # Combinar todas as características em um único vetor de comprimento fixo
    caracteristicas = [homogeneidade] + list(descritores_de_cor)
    
    return caracteristicas

# Verificar consistência no comprimento dos vetores de características
imagens_puras = glob.glob(r'E:\projetos\imagensTcc\pure\*.jpg')
imagens_impuras = glob.glob(r'E:\projetos\imagensTcc\impure\*.jpg')

caracteristicas = []
rotulos = []

for caminho_imagem in imagens_puras:
    imagem = cv2.imread(caminho_imagem)
    caracteristicas.append(extrair_caracteristicas(imagem))
    rotulos.append(1)

for caminho_imagem in imagens_impuras:
    imagem = cv2.imread(caminho_imagem)
    caracteristicas.append(extrair_caracteristicas(imagem))
    rotulos.append(0)

lengths = [len(c) for c in caracteristicas]
if len(set(lengths)) != 1:
    print("Erro: As características extraídas têm comprimentos diferentes:")
    for i, l in enumerate(lengths):
        print(f"Imagem {i}: comprimento = {l}")
else:
    X = np.array(caracteristicas)
    y = np.array(rotulos)

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = modelo_rf.predict(X_test)
    precisao_rf = accuracy_score(y_test, y_pred)
    print(f'Acurácia do Random Forest: {precisao_rf:.2f}')

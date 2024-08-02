import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurar a codificação do stdout para UTF-8
sys.stdout.reconfigure(encoding='UTF-8')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    imagens = []
    rotulos = []

    for caminho_imagem in imagens_puras:
        imagem = cv2.imread(caminho_imagem)
        imagem = cv2.resize(imagem, (128, 128))
        imagens.append(imagem)
        rotulos.append(1)

    for caminho_imagem in imagens_impuras:
        imagem = cv2.imread(caminho_imagem)
        imagem = cv2.resize(imagem, (128, 128))
        imagens.append(imagem)
        rotulos.append(0)

    X = np.array(imagens)
    y = np.array(rotulos)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Construir o modelo CNN
    modelo_cnn = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])

    modelo_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    modelo_cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    precisao_cnn = modelo_cnn.evaluate(X_test, y_test, verbose=0)[1]
    print(f'Acurácia da CNN: {precisao_cnn:.2f}')

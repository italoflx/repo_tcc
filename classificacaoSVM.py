import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import glob
from utils import *

def calcular_homogeneidade(imagem):
    glcm = graycomatrix(imagem, distances=[1], angles=[0], symmetric=True, normed=True)
    homogeneidade = graycoprops(glcm, 'homogeneity')[0, 0]
    return homogeneidade

def extrair_caracteristicas(imagem):
    imagem_cmyk = bgr_to_cmyk(imagem)
    _, _, y, k = cv2.split(imagem_cmyk)
    
    _, imagem_binaria = cv2.threshold(y, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    imagem_binaria = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)
    
    homogeneidade = calcular_homogeneidade(imagem_binaria)
    
    return [homogeneidade]

imagens_puras = glob.glob(r'E:\projetos\imagensTcc\pure\*.jpg')
imagens_impuras = glob.glob(r'E:\projetos\imagensTcc\impure\*.jpg')

caracteristicas = []
rotulos = []

print(caracteristicas)

for caminho_imagem in imagens_puras:
    imagem = cv2.imread(caminho_imagem)
    caracteristicas.append(extrair_caracteristicas(imagem))
    rotulos.append(1) 

# Processar imagens impuras
for caminho_imagem in imagens_impuras:
    imagem = cv2.imread(caminho_imagem)
    caracteristicas.append(extrair_caracteristicas(imagem))
    rotulos.append(0) 

X = np.array(caracteristicas)
y = np.array(rotulos)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = SVC(kernel='linear')
modelo.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

precisao = accuracy_score(y_test, y_pred)
print(f'Acuracia: {precisao:.2f}')
# 56% de acuracia atualmente 25/07/2024 03:52AM

# nova_imagem = cv2.imread(r'E:\projetos\imagensTcc\CamaraoAmarelo6x1.jpg')
# caracteristicas_nova_imagem = np.array(extrair_caracteristicas(nova_imagem)).reshape(1, -1)
# classe_predita = modelo.predict(caracteristicas_nova_imagem)
# print(f'Classe predita: {"Puro" if classe_predita[0] == 1 else "Impuro"}')


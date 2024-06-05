import cv2
import numpy as np
from utils import redimensionar_para_tela, passa_alta, alargar_contraste

imagem = cv2.imread(r'E:\projetos\imagensTcc\MotoG10_Finais\CamaraoVermelho3x1.jpg')

b, g, r = cv2.split(imagem)

# Defino os limiares, depois declaro uma mascara e aplico nos pixels respectivos 255
# mask = (r > 127) & (g < 100) & (b < 100)

# Mascara = np.zeros_like(r, dtype=np.uint8)
# Mascara[mask] = 255

# r[(r > 100)] = 0
r[(r > 95)] = 0
g[(r < 50)] = 0
b[(r < 50)] = 0

# NAO TA FUNCIONANDO
mask = (r > 20)

# Converter a máscara booleana para uma imagem binária (0 e 255)
binary_image = np.zeros_like(r, dtype=np.uint8)
binary_image[mask] = 255

imagem_modificada = cv2.merge((b, g, r))

#A PARTIR DAQUI 
# 1) PEGAR O MAIOR OBJETO
# 2) PREENCHER O MEIO
# 3) OBTER A MELHOR MASCARA COM ESSE RESULTADO

imagem_cinza = cv2.cvtColor(imagem_modificada, cv2.COLOR_BGR2GRAY)

imagem_binaria = cv2.adaptiveThreshold(imagem_cinza, 255, 
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

maior_contorno = max(contornos, key=cv2.contourArea)

mascara_maior_contorno = np.zeros_like(imagem_binaria)
cv2.drawContours(mascara_maior_contorno, [maior_contorno], -1, (255), thickness=cv2.FILLED)

#NAO ESTA FUNCIONANDO, DESCONFIO DO ERRO 'Invalid SOS parameters for sequential JPEG'
kernel = np.ones((10, 10), np.uint8)
mascara_erodida = cv2.erode(mascara_maior_contorno, kernel, iterations=2)

imagem_mascarada = cv2.bitwise_and(imagem, imagem, mask=mascara_maior_contorno)

cv2.imshow('Imagem Original', redimensionar_para_tela(imagem))
#A VARIAVEL IMAGEM MODIFICADA, ESTA APENAS FILTRANDO O VERMELHO, SEM QUALQUER PROCESSAMENTO
cv2.imshow('Imagem Modificada', redimensionar_para_tela(imagem_modificada))
cv2.imshow('Imagem Mascara', redimensionar_para_tela(binary_image))
#A VARIAVEL IMAGEM MASCARADA, ESTA COM OS PROCESSOS: OBTER CONTORNOS, DESENHAR CONTORNOS, DILATACAO, EROSAO
cv2.imshow('Imagem Teste', redimensionar_para_tela(imagem_mascarada))

cv2.waitKey(0)
cv2.destroyAllWindows()

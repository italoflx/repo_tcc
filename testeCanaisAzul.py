import cv2
import numpy as np
from utils import redimensionar_para_tela, bgr_to_cmyk

imagem = cv2.imread(r'E:\projetos\imagensTcc\MotoG10_Finais\CamaraoAzul1x1.jpg')
cv2.imshow('original', redimensionar_para_tela(imagem))

imagem_cmyk = bgr_to_cmyk(imagem)

c, m, y, k = cv2.split(imagem_cmyk)

imagem_cinza = cv2.cvtColor(imagem_cmyk, cv2.COLOR_BGR2GRAY)

_, imagem_binaria = cv2.threshold(k, 180, 255, cv2.THRESH_BINARY)

contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contorno in contornos:
    cv2.drawContours(imagem_binaria, [contorno], 0, 255, -1)

kernel = np.ones((5, 5), np.uint8)
imagem_binaria = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)

imagem_mascarada_final = cv2.bitwise_and(imagem, imagem, mask=imagem_binaria)

cv2.imshow('canal azul', redimensionar_para_tela(k))
cv2.imshow('imagem binaria', redimensionar_para_tela(imagem_binaria))
cv2.imshow('imagem final', redimensionar_para_tela(imagem_mascarada_final))

cv2.waitKey(0)
cv2.destroyAllWindows()

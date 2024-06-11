import cv2
import numpy as np
from utils import redimensionar_para_tela, alargar_contraste, passa_alta, bgr_to_cmyk

imagem = cv2.imread(r'E:\projetos\imagensTcc\MotoG10_Finais\CamaraoAmarelo4x1.jpg')
cv2.imshow('original', redimensionar_para_tela(imagem))

imagem_cmyk = bgr_to_cmyk(imagem)

c, m, y, k = cv2.split(imagem_cmyk)

imagem_cinza = cv2.cvtColor(imagem_cmyk, cv2.COLOR_BGR2GRAY)

cv2.imshow('imagem cinza', redimensionar_para_tela(imagem_cinza))

_, imagem_binaria = cv2.threshold(y, 128, 255, cv2.THRESH_BINARY)

cv2.imshow('imagem binaria', redimensionar_para_tela(imagem_binaria))

edges = cv2.Canny(imagem_binaria, 100, 200)

imagem_mascarada_final = cv2.bitwise_and(imagem, imagem, mask=imagem_binaria)

cv2.imshow('imagem edges', redimensionar_para_tela(edges))
cv2.imshow('imagem final', redimensionar_para_tela(imagem_mascarada_final))
cv2.imshow('y', redimensionar_para_tela(y))
cv2.waitKey(0)
cv2.destroyAllWindows()
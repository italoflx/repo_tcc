import cv2
import numpy as np
from utils import redimensionar_para_tela, alargar_contraste, passa_alta

imagem = cv2.imread(r'E:\projetos\imagensTcc\MotoG10_Finais\CamaraoVerde1x1.jpg')

imagem_escolhida = imagem

b, g, r = cv2.split(imagem)

r[(r < 40)] = 0
g[(r > 40)] = 0
b[(r < 90)] = 0

mask = (r > 20)
binary_image = np.zeros_like(r, dtype=np.uint8)
binary_image[mask] = 255

imagem_modificada = cv2.merge((b, g, r))

imagem_cinza = cv2.cvtColor(imagem_modificada, cv2.COLOR_BGR2GRAY)

imagem_binaria = cv2.adaptiveThreshold(imagem_cinza, 255, 
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contornos:
    maior_contorno = max(contornos, key=cv2.contourArea)

    mascara_maior_contorno = np.zeros_like(imagem_binaria)
    cv2.drawContours(mascara_maior_contorno, [maior_contorno], -1, (255), thickness=cv2.FILLED)

    if mascara_maior_contorno.shape != imagem_escolhida.shape[:2]:
        mascara_maior_contorno = cv2.resize(mascara_maior_contorno, (imagem_escolhida.shape[1], imagem_escolhida.shape[0]))

    cv2.imshow('Mascara Maior Contorno', redimensionar_para_tela(mascara_maior_contorno))

    kernel = np.ones((8, 8), np.uint8)
    mascara_erodida = cv2.erode(mascara_maior_contorno, kernel, iterations=1)

    mascara_erodida = mascara_erodida.astype(np.uint8)

    imagem_mascarada = cv2.bitwise_and(imagem_escolhida, imagem_escolhida, mask=mascara_erodida)
    
    cv2.imshow('Imagem Mascarada', redimensionar_para_tela(imagem_mascarada))
else:
    print("Nenhum contorno encontrado.")

cv2.imshow('Imagem Original', redimensionar_para_tela(imagem_escolhida))
cv2.imshow('Imagem Mascara', redimensionar_para_tela(binary_image))

cv2.waitKey(0)
cv2.destroyAllWindows()

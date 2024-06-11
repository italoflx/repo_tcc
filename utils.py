import cv2
import numpy as np
from screeninfo import get_monitors

def redimensionar_para_tela(imagem):
    monitor = get_monitors()[0]
    largura_tela = monitor.width
    altura_tela = monitor.height
    
    altura_imagem, largura_imagem = imagem.shape[:2]
    
    escala_largura = largura_tela / largura_imagem
    escala_altura = altura_tela / altura_imagem
    escala = min(escala_largura, escala_altura)
    
    nova_largura = int(largura_imagem * escala)
    nova_altura = int(altura_imagem * escala)
    
    imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return imagem_redimensionada

def alargar_contraste(imagem):
    fator = 2.3
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * fator, 0, 255)

    imagem_contraste = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return imagem_contraste

def passa_alta(imagem):
    kernel_passa_alta = np.array([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]])
    
    imagem_passa_alta = cv2.filter2D(imagem, -1, kernel_passa_alta)
    return imagem_passa_alta

def bgr_to_cmyk(image):
    b, g, r = cv2.split(image)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    k = 1 - np.max([r, g, b], axis=0)
    c = (1 - r - k) / (1 - k + 1e-10)
    m = (1 - g - k) / (1 - k + 1e-10)
    y = (1 - b - k) / (1 - k + 1e-10)
    
    cmyk = (np.dstack((c, m, y, k)) * 255).astype(np.uint8)
    return cmyk
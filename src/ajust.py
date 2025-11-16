import cv2 as cv
import numpy as np

# --- 1. Variáveis Globais e Inicialização ---

# Carregar e redimensionar a imagem
img_original = cv.imread("assets\Image_Colors.png")
if img_original is None:
    print("ERRO: Imagem 'Image_Colors.png' não encontrada.")
    exit()

rate = 2
h, w = img_original.shape[:2]
# Redimensionamento
img_original = cv.resize(img_original, (w // rate, h // rate)) 

# CONVERSÃO INICIAL PARA HSV E ESCALA DE CINZA
img_hsv = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)


# Nome da janela de controle e resultados
WINDOW_NAME = "Controle de Segmentacao"
RESULT_WINDOW = "Resultado da Segmentacao"

# Cria a janela e as trackbars
cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL) # Permite redimensionar a janela de controle
cv.namedWindow(RESULT_WINDOW)

# --- TRACKBARS INRANGE (H, S, V) ---
# HUE (Matiz) vai até 179 no OpenCV
cv.createTrackbar("H_Min", WINDOW_NAME, 0, 179, lambda x: None)
cv.createTrackbar("S_Min", WINDOW_NAME, 50, 255, lambda x: None)
cv.createTrackbar("V_Min", WINDOW_NAME, 50, 255, lambda x: None)

cv.createTrackbar("H_Max", WINDOW_NAME, 179, 179, lambda x: None)
cv.createTrackbar("S_Max", WINDOW_NAME, 255, 255, lambda x: None)
cv.createTrackbar("V_Max", WINDOW_NAME, 255, 255, lambda x: None)

# --- TRACKBARS THRESHOLD & CANNY ---
cv.createTrackbar("Threshold Valor", WINDOW_NAME, 200, 255, lambda x: None)
cv.createTrackbar("Canny Limiar 1", WINDOW_NAME, 70, 255, lambda x: None)
cv.createTrackbar("Canny Limiar 2", WINDOW_NAME, 150, 255, lambda x: None)

# --- 2. Função Auxiliar de Centro ---
def get_central_point(objecto):
    """Calcula o centroide de um contorno."""
    M = cv.moments(objecto)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None, None

# --- 3. Função Principal de Processamento ---
def processar_e_mostrar(img_hsv, img_original):
    """Lê os trackbars, processa a imagem e exibe o resultado."""
    
    # 3.1. Leitura dos Parâmetros do INRANGE
    h_min = cv.getTrackbarPos("H_Min", WINDOW_NAME)
    s_min = cv.getTrackbarPos("S_Min", WINDOW_NAME)
    v_min = cv.getTrackbarPos("V_Min", WINDOW_NAME)
    h_max = cv.getTrackbarPos("H_Max", WINDOW_NAME)
    s_max = cv.getTrackbarPos("S_Max", WINDOW_NAME)
    v_max = cv.getTrackbarPos("V_Max", WINDOW_NAME)
    
    # Leitura dos Parâmetros de Threshold & Canny
    thresh_val = cv.getTrackbarPos("Threshold Valor", WINDOW_NAME)
    canny_t1 = cv.getTrackbarPos("Canny Limiar 1", WINDOW_NAME)
    canny_t2 = cv.getTrackbarPos("Canny Limiar 2", WINDOW_NAME)
    
    # Garante que T1 < T2
    if canny_t1 >= canny_t2:
        canny_t2 = canny_t1 + 1 
        cv.setTrackbarPos("Canny Limiar 2", WINDOW_NAME, canny_t2)

    # 3.2. APLICAÇÃO DO INRANGE (CRIAÇÃO DA MÁSCARA)
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    
    # Mascara a imagem original em HSV
    mask_inrange = cv.inRange(img_hsv, lower_bound, upper_bound)
    
    # 3.3. FILTROS MORFOLÓGICOS (para limpeza da máscara)
    # Kernel 3x3 é bom para remover ruído da máscara
    kernel_morf = np.ones((3, 3), np.uint8)
    mask_cleaned = cv.morphologyEx(mask_inrange, cv.MORPH_OPEN, kernel_morf)
    
    # 3.4. APLICAÇÃO DO CANNY (na máscara limpa)
    # A máscara (mask_cleaned) é binária, então aplicamos o Canny diretamente nela.
    edges = cv.Canny(mask_cleaned, canny_t1, canny_t2)
    
    # 3.5. Encontrar Contornos
    objetos, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Cria uma cópia da imagem original para desenhar os contornos
    img_output = img_original.copy()
    
    # 3.6. Desenhar Contornos e Centros
    objetos_validos = 0
    for objeto in objetos:
        # Pula contornos muito pequenos (ruído)
        if cv.contourArea(objeto) < 50:
            continue
            
        objetos_validos += 1
        
        # Calcular centro
        cx, cy = get_central_point(objeto)
        
        # Desenhar
        # Desenha o contorno atual 'objeto' na cor verde
        cv.drawContours(img_output, [objeto], -1, (0, 255, 0), 2) 
        if cx is not None:
            cv.circle(img_output, (cx, cy), 5, (0, 0, 255), -1) # Centro em Vermelho
    
    # 3.7. Exibir Resultados
    cv.putText(img_output, f"Objetos: {objetos_validos}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar as janelas
    cv.imshow(RESULT_WINDOW, img_output)
    cv.imshow("Mascara HSV (inRange)", mask_inrange)
    cv.imshow("Contornos Canny (input)", edges)

# --- 4. Loop de Execução Interativa ---
while True:
    processar_e_mostrar(img_hsv, img_original)
    
    key = cv.waitKey(100)
    if key == ord('q'):
        break

cv.destroyAllWindows()
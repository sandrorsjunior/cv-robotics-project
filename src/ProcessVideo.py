import cv2
import numpy as np
from ProcessImage import ProcessImage


def detect_red_color_webcam():
    processImage = ProcessImage()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    red_color_limits = processImage.calculate_hsv_bounds((70, 82, 231))

    print("🔴 Pressione 'q' para sair da aplicação.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        red_mask = processImage.segmentation_by_color(red_color_limits, frame)

        clean_mask = processImage.remove_noise(red_mask)

        red_segment = cv2.bitwise_and(frame, frame, mask=clean_mask)
        
        cv2.imshow('Detecção de Vermelho', red_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_red_color_webcam()
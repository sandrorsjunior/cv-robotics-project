import cv2
import numpy as np
from ProcessImage import ProcessImage


def detect_colors_webcam(colors_to_detect, sigma=50):
    """
    Detecta uma lista de cores a partir da webcam.
    :param colors_to_detect: Uma lista de tuplas de cores em BGR. Ex: [(B1,G1,R1), (B2,G2,R2)]
    """
    processImage = ProcessImage()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    all_color_limits = []
    for color_bgr in colors_to_detect:
        limits = processImage.calculate_hsv_bounds(color_bgr, sigma=sigma)
        all_color_limits.extend(limits)

    print("Pressione 'q' para sair da aplicação.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combined_mask = processImage.segmentation_by_color(all_color_limits, frame)

        clean_mask = processImage.remove_noise(combined_mask)

        color_segment = cv2.bitwise_and(frame, frame, mask=clean_mask)

        cv2.imshow('Mascara Combinada', color_segment)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Defina aqui a lista de cores (em BGR) que você quer detectar
    cores_alvo = [
        (0, 0, 255),  # Vermelho/Laranja que você estava usando
        (0, 255, 0),  # Verde puro
        (255, 0, 0)   # Azul puro
    ]
    detect_colors_webcam(cores_alvo)

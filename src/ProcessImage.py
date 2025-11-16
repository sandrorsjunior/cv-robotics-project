import cv2
import numpy as np

class ProcessImage:
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._load_image()
        self.h, self.w = self.image.shape[:2]
        self.rate = 1


    def _load_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileExistsError(f"It was impossible load the image from the path: {self.image_path}")
        return img

    def convert_to_grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def convert_to_hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    @staticmethod
    def save_image(filename, img):
        cv2.imwrite(filename, img)
        print(f" -> Imagem '{filename}' salva.")

    def apply_color_mask(self, hsv_image, lower_bound, upper_bound):
        if hsv_image is not None:
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            return cv2.bitwise_and(self.image, self.image, mask=mask)
        return None

    def apply_morphological_opening(self, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

    def apply_morphological_closing(self, kernel_size=(3, 3)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

    def apply_morphological_gradient(self, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8) 
        return cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)
    
    def remove_noise(self, mascara, kernel = np.ones((5,5), np.uint8)):
        mascara = cv2.dilate(mascara, kernel)
        mascara = cv2.erode(mascara, kernel)
        return mascara
    
    def segmentation_by_color(self, color_range, img):
        img_HSV = self.convert_to_hsv(img)
        mask = cv2.inRange(img_HSV, color_range[0], color_range[1])
        return mask
    
    def map_distinct_colors(self, background_color):
        colors = set()
        img = cv2.resize(self.image, (self.h//self.rate, self.w//self.rate))
        mask = self.segmentation_by_color(background_color, self.image)
        mask = self.remove_noise(mask)
        T, bin = cv2.threshold(mask, 234, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(bin, 70, 150)
        objetos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_split = cv2.split(img)

        for objeto in objetos:
            cx, cy = self.get_central_point(objeto)
            color = self.get_color_of_object(img_split, (cy,cx))
            colors.add(color)

        return list(colors)

    
    def get_central_point(self, objecto):
        M = cv2.moments(objecto)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy

        
    def get_color_of_object(self, img_chanels, central_pixel):
        linha_y, coluna_x = central_pixel 
        chanel_b = int(img_chanels[0][linha_y, coluna_x])
        chanel_g = int(img_chanels[1][linha_y, coluna_x])
        chanel_r = int(img_chanels[2][linha_y, coluna_x])
        
        return (chanel_b, chanel_g, chanel_r)
    
    def calculate_hsv_bounds(self, bgr_color_tuple, sigma=15):
        
        B, G, R = bgr_color_tuple
        bgr_array = np.uint8([[[B, G, R]]])
        hsv_array = self.convert_to_hsv(bgr_array)
        
        H, S, V = hsv_array[0, 0]

        sigma_H = sigma // 2 
        sigma_SV = sigma
        
        h_min = np.clip(int(H) - sigma_H, 0, 179)
        s_min = np.clip(int(S) - sigma_SV, 0, 255)
        v_min = np.clip(int(V) - sigma_SV, 0, 255)
        
        # --- Cálculo dos Limites Superior (top) ---
        
        h_max = np.clip(int(H) + sigma_H, 0, 179)
        s_max = np.clip(int(S) + sigma_SV, 0, 255)
        v_max = np.clip(int(V) + sigma_SV, 0, 255)
        
        down = (int(h_min), int(s_min), int(v_min))
        top = (int(h_max), int(s_max), int(v_max))
        
        return [down, top]

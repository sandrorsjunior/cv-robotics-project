import cv2
import numpy as np




if __name__ == "__main__":
    from ProcessImage import ProcessImage
    
    background_color = (np.array([0,7,50]), np.array([179, 255, 255]))

    processImage = ProcessImage("assets/Image_Colors.png")
    list_of_colors = processImage.map_distinct_colors(background_color)
    for i, color in enumerate(list_of_colors):
        limit = processImage.calculate_hsv_bounds(list_of_colors[i])
        mask = processImage.segmentation_by_color(limit, processImage.image)
        img = cv2.bitwise_and(processImage.image, processImage.image, mask=mask)
        ProcessImage.save_image(f"./tasks/ex1/color_{i}.jpg", img)
        

    


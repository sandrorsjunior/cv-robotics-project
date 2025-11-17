import cv2
import numpy as np


if __name__ == "__main__":
    from ProcessImage import ProcessImage
    
    processImage = ProcessImage("assets/Image_Colors.png")

    #*******************************************************************
    #                               TASK 1
    #*******************************************************************

    img_HSV = processImage.convert_to_hsv(processImage.image)
    ProcessImage.show_image("HSV", img_HSV)
    ProcessImage.save_image("./tasks/ex1/HSV.jpg", img_HSV)



    #*******************************************************************
    #                               TASK 2
    #*******************************************************************

    background_color_range = [[np.array([0, 7, 50]), np.array([179, 255, 255])]]
    list_of_colors = processImage.map_distinct_colors(background_color_range)
    for i, color in enumerate(list_of_colors):
        limit = processImage.calculate_hsv_bounds(list_of_colors[i])
        print(limit)
        mask = processImage.segmentation_by_color(limit, processImage.image)
        img = cv2.bitwise_and(processImage.image, processImage.image, mask=mask)
        ProcessImage.save_image(f"./tasks/ex2/color_{i}.jpg", img)
        

    
    #*******************************************************************
    #                               TASK 3
    #*******************************************************************

    processImage_BinaryImg = ProcessImage("assets/BinaryImg.png")
    #img_no_noise = processImage_BinaryImg.convert_to_grayscale(processImage_BinaryImg.image)
    img_no_noise = processImage_BinaryImg.remove_noise(processImage_BinaryImg.image,
                                                       kernel_dilate=np.ones((2,2), np.uint8), 
                                                       kernel_erode=np.ones((3,3), np.uint8))

    ProcessImage.save_image("./tasks/ex3/BinaryImg_processed.jpg", img_no_noise)


    #*******************************************************************
    #                               TASK 4
    #*******************************************************************
    processImage_head = ProcessImage("assets/head_gaussian_noise.png")
    
    # Converte para escala de cinza, onde as operações morfológicas para remoção de ruído são mais eficazes
    gray_head_image = processImage_head.convert_to_grayscale(processImage_head.image)
    ProcessImage.save_image("./tasks/ex4/head_original_gray.png", gray_head_image)

    # Experimenta diferentes tamanhos de kernel
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    print("Analisando o efeito de diferentes tamanhos de kernel:")

    for size in kernel_sizes:
        print(f" -> Aplicando kernel de tamanho {size}...")
        smoothed_img = processImage_head.smooth_with_morphology(gray_head_image, kernel_size=size)
        filename = f"./tasks/ex4/head_smoothed_kernel_{size[0]}x{size[1]}.jpg"
        ProcessImage.save_image(filename, smoothed_img)

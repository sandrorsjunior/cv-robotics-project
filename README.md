# 🤖 Projeto de Visão Computacional (Robótica)

Este projeto foi desenvolvido como um trabalho prático para a disciplina
de Visão Computacional do curso de Robótica. O objetivo principal é
demonstrar o domínio de conceitos fundamentais como conversão de cores,
segmentação, morfologia matemática e processamento de vídeo em tempo
real, utilizando a biblioteca OpenCV.

O design do código foca na modularidade, com a classe **ProcessImage**
atuando como um "cérebro" de visão computacional, separando a lógica de
processamento dos scripts de execução (`main.py` e `ProcessVideo.py`).

------------------------------------------------------------------------

## 📂 Estrutura do Projeto

A organização segue um padrão limpo e funcional:

    opencv/
    │
    ├── assets/                 # Imagens de entrada (Image_Colors.png, BinaryImg.png, etc.)
    │
    ├── src/                    # Código-fonte principal
    │   ├── ProcessImage.py     # CLASSE: Lógica de Visão Computacional
    │   ├── main.py             # SCRIPT: Executa tarefas estáticas (Questões 1-5)
    │   ├── ProcessVideo.py     # SCRIPT: Executa a detecção em tempo real (Questão 6)
    │   └── ajust.py            # AUXILIAR: Ferramenta de calibração HSV (Trackbars)
    │
    ├── tasks/                  # Resultados das tarefas (tasks/ex1, tasks/ex2, etc.)
    │
    ├── poetry.lock             # Arquivo de bloqueio de dependências
    └── pyproject.toml          # Configuração de dependências (Poetry)

------------------------------------------------------------------------

## 🛠️ Configuração e Execução

O projeto utiliza **Poetry** para garantir um ambiente de execução
consistente.

### 1. Instalação das Dependências

Certifique-se de ter o Python e o Poetry instalados. Na raiz do projeto,
execute:

    poetry install

### 2. Execução das Tarefas Estáticas (Questões 1 a 5)

Este script processa todas as imagens da pasta `assets/` e salva os
resultados em `tasks/`.

    poetry run python src/main.py

### 3. Execução em Tempo Real (Questão 6)

Este script ativa a webcam para detecção de cores RGB em tempo real:

    poetry run python src/ProcessVideo.py

Pressione **'q'** para fechar a janela.

------------------------------------------------------------------------

## 🧠 Detalhamento da Classe `src/ProcessImage.py`

Esta classe contém toda a lógica complexa e reutilizável, garantindo que
os scripts principais sejam concisos.

### 🔄 Funções de Conversão e I/O

  -------------------------------------------------------------------------
  Função                        Descrição
  ----------------------------- -------------------------------------------
  `convert_to_grayscale(img)`   Converte a imagem BGR para Escala de Cinza
                                (`cv2.COLOR_BGR2GRAY`).

  `convert_to_hsv(img)`         Converte a imagem BGR para o espaço de
                                cores HSV.

  `save_image(filename, img)`   Função estática para salvar qualquer imagem
                                processada no disco.
  -------------------------------------------------------------------------

### 🎨 Funções de Segmentação e Análise de Cores

  ---------------------------------------------------------------------------------------------
  Função                                        Descrição                        Uso
  --------------------------------------------- -------------------------------- --------------
  `calculate_hsv_bounds(bgr_color, sigma=15)`   Calcula intervalos HSV robustos, Q2, Q6
                                                tratando o wrap-around do Hue.   

  `segmentation_by_color(color_range, img)`     Aplica limites HSV para obter a  Q2, Q6
                                                máscara binária da cor alvo.     

  `map_distinct_colors(background_color)`       Identifica cores distintas dos   Q2
                                                objetos via contornos e          
                                                momentos.                        
  ---------------------------------------------------------------------------------------------

### 🧩 Funções de Morfologia Matemática

  ------------------------------------------------------------------------------------------------------
  Função                                                 Descrição                        Uso
  ------------------------------------------------------ -------------------------------- --------------
  `remove_noise(mascara, kernel_dilate, kernel_erode)`   Fechamento morfológico (remove   Q3, Q6
                                                         buracos e ruídos).               

  `smooth_with_morphology(img, kernel_size)`             Suavização morfológica contra    Q4
                                                         ruído sal-e-pimenta.             

  `apply_morphological_gradient(...)`                    Calcula o gradiente morfológico  Q5
                                                         para realce de bordas.           
  ------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📝 Implementação das Questões

### **Questão 1 --- Conversão de Cor**

-   Conversão de BGR → HSV usando `convert_to_hsv`.
-   Saída: `tasks/ex1/HSV.jpg`

### **Questão 2 --- Separação de Cores**

-   Segmentação das cores da imagem usando `map_distinct_colors`,
    `calculate_hsv_bounds` e `segmentation_by_color`.
-   Saída: `tasks/ex2/color_*.jpg`

### **Questão 3 --- Morfologia (Limpeza Binária)**

-   Fechamento morfológico com `remove_noise`.
-   Saída: `tasks/ex3/BinaryImg_processed.jpg`

### **Questão 4 --- Suavização Morfológica**

-   Suavização contra ruído Gaussiano em diferentes kernels.
-   Saída: `tasks/ex4/head_smoothed_kernel_*.jpg`

### **Questão 5 --- Detecção de Contornos**

-   Uso do gradiente morfológico e `cv2.findContours`.
-   Aplicado sobre imagens da Q2.

### **Questão 6 --- Separação de Cores em Tempo Real**

-   Webcam + segmentação contínua com `calculate_hsv_bounds`,
    `segmentation_by_color` e `remove_noise`.

------------------------------------------------------------------------

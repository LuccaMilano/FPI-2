import cv2
import tkinter as tk
import numpy as np
from matplotlib import pyplot as plot



class interface(tk.Frame):
    # Definições que serão utilizadas no decorrer do código
    def __init__(self, img_copy, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        self.img = img_copy
        self.soma = 0


    # Cria todos os objetos interativos da interface
    def create_widgets(self):

        # Botão que mostra a imagem original
        self.button0 = tk.Button(self)
        self.button0["text"] = "Copy Original Image"
        self.button0["command"] = self.copy_original
        self.button0.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que mostra a imagem atual
        self.button1 = tk.Button(self)
        self.button1["text"] = "Show Current Image"
        self.button1["command"] = self.show_current
        self.button1.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que realiza a rotação horizontal da imagem
        self.button2 = tk.Button(self)
        self.button2["text"] = "Rotate Horizontal"
        self.button2["command"] = self.rotation_horizontal
        self.button2.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que realiza a rotação vertical da imagem
        self.button3 = tk.Button(self)
        self.button3["text"] = "Rotate Vertical"
        self.button3["command"] = self.rotation_vertical
        self.button3.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que converte uma imagem colorida para tons de cinza
        self.button4 = tk.Button(self)
        self.button4["text"] = "Convert to gray"
        self.button4["command"] = self.convert_to_gray
        self.button4.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que disponibiliza a opção de salvar a imagem modificada
        self.button5 = tk.Button(self)
        self.button5["text"] = "Save Current Image"
        self.button5["command"] = self.save
        self.button5.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que faz a quantização de tons de acordo com o número digitado
        self.button6 = tk.Button(self)
        self.button6["text"] = "Quantize Colors"
        self.button6["command"] = self.Verify_tones_number
        self.button6.pack(side="top")

        # Label de informação do campo em branco em que deve ser digitado o número de tons
        self.nameLabel = tk.Label(self, text="Coloros:", font=("Arial", "12"))
        self.nameLabel.pack(side='top')

        # Recebe o número de tons que quantizarão a imagem
        self.tones = tk.Entry(self)
        self.tones["width"] = 30
        self.tones["font"] = ("Arial", "12")
        self.tones.pack(side='top')
        # ----------------------------------------------------------------------

        self.button7 = tk.Button(self)
        self.button7["text"] = "Rotate Counterclockwise"
        self.button7["command"] = self.Rotate_90_antih
        self.button7.pack(side="top")
        # ----------------------------------------------------------------------

        self.button8 = tk.Button(self)
        self.button8["text"] = 'Rotate Clockwise'
        self.button8["command"] = self.Rotate_90_h
        self.button8.pack(side="top")
        # ----------------------------------------------------------------------

        self.button9 = tk.Button(self)
        self.button9["text"] = 'Plot Histogram'
        self.button9["command"] = self.Plot_Histogram
        self.button9.pack(side="top")
        # ----------------------------------------------------------------------
        self.button10 = tk.Button(self)
        self.button10["text"] = 'Adjust Brightness'
        self.button10["command"] = self.Adjust_Brightness
        self.button10.pack(side="top")

        # Label de informação do campo em branco em que deve ser digitado o aumento\diminuição de brilho
        self.nameLabel2 = tk.Label(self, text="Brightness:", font=("Arial", "12"))
        self.nameLabel2.pack(side='top')

        # Recebe o brilho modificara a imagem
        self.brightness = tk.Entry(self)
        self.brightness["width"] = 30
        self.brightness["font"] = ("Arial", "12")
        self.brightness.pack(side='top')
        # ----------------------------------------------------------------------

        self.button11 = tk.Button(self)
        self.button11["text"] = 'Adjust Contrast'
        self.button11["command"] = self.Adjust_Contrast
        self.button11.pack(side="top")

        # Label de informação do campo em branco em que deve ser digitado o contraste
        self.nameLabel3 = tk.Label(self, text="Contrast:", font=("Arial", "12"))
        self.nameLabel3.pack(side='top')

        # Recebe o número do contraste que modificara a imagem
        self.contrast = tk.Entry(self)
        self.contrast["width"] = 30
        self.contrast["font"] = ("Arial", "12")
        self.contrast.pack(side='top')
        # ----------------------------------------------------------------------

        self.button12 = tk.Button(self)
        self.button12["text"] = 'Calculate Negative'
        self.button12["command"] = self.Negative_Calculation
        self.button12.pack(side="top")
        # ----------------------------------------------------------------------
        self.button13 = tk.Button(self)
        self.button13["text"] = 'Equalize Image'
        self.button13["command"] = self.Equalize_Image
        self.button13.pack(side="top")
        # ----------------------------------------------------------------------
        self.button14 = tk.Button(self)
        self.button14["text"] = 'Zoom In Image'
        self.button14["command"] = self.ZoomIn_Image
        self.button14.pack(side="top")
        # ----------------------------------------------------------------------

        self.button15 = tk.Button(self)
        self.button15["text"] = 'Zoom Out Image'
        self.button15["command"] = self.ZoomOut_Image
        self.button15.pack(side="top")
        # ----------------------------------------------------------------------
        self.button16 = tk.Button(self)
        self.button16["text"] = 'Convolution Gaussiano'
        self.button16["command"] = self.Convolution_Gaussiano
        self.button16.pack(side="top")
        # ----------------------------------------------------------------------
        self.button17 = tk.Button(self)
        self.button17["text"] = 'Convolution Laplace'
        self.button17["command"] = self.Convolution_Laplaciano
        self.button17.pack(side="top")
        # ----------------------------------------------------------------------
        self.button18 = tk.Button(self)
        self.button18["text"] = 'Convolution Altas Genérico'
        self.button18["command"] = self.Convolution_PassaAlta
        self.button18.pack(side="top")
        # ----------------------------------------------------------------------
        self.button19 = tk.Button(self)
        self.button19["text"] = 'Convolution Prewitt Hx'
        self.button19["command"] = self.Convolution_PrewittHx
        self.button19.pack(side="top")
        # ----------------------------------------------------------------------
        self.button20 = tk.Button(self)
        self.button20["text"] = 'Convolution Prewitt Hy'
        self.button20["command"] = self.Convolution_PrewittHy
        self.button20.pack(side="top")
        # ----------------------------------------------------------------------
        self.button21 = tk.Button(self)
        self.button21["text"] = 'Convolution Sobel Hx'
        self.button21["command"] = self.Convolution_SobelHx
        self.button21.pack(side="top")
        # ----------------------------------------------------------------------
        self.button22 = tk.Button(self)
        self.button22["text"] = 'Convolution Sobel Hy'
        self.button22["command"] = self.Convolution_SobelHy
        self.button22.pack(side="top")
        # ----------------------------------------------------------------------

        # Botão que encerra o programa
        self.quit = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="bottom")
        # ----------------------------------------------------------------------

    # Realiza a convolução da imagem segundo filtro SobelHy
    def Convolution_SobelHy(self):
        cv2.destroyAllWindows()
        kernel = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        self.Convolution(kernel, 1)
        cv2.imshow('SobelHy Filter Image', self.img)

    # Realiza a convolução da imagem segundo filtro SobelHx
    def Convolution_SobelHx(self):
        cv2.destroyAllWindows()
        kernel = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        self.Convolution(kernel, 1)
        cv2.imshow('SobelHx Filter Image', self.img)

    # Realiza a convolução da imagem segundo filtro PrewittHy
    def Convolution_PrewittHy(self):
        cv2.destroyAllWindows()
        kernel = [[-1, -1, -1],
                  [0, 0, 0],
                  [1, 1, 1]]
        self.Convolution(kernel, 1)
        cv2.imshow('PrewittHy Filter Image', self.img)

    # Realiza a convolução da imagem segundo filtro PrewittHX
    def Convolution_PrewittHx(self):
        cv2.destroyAllWindows()
        kernel = [[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]]
        self.Convolution(kernel, 1)
        cv2.imshow('PrewittHx Filter Image', self.img)


    # Realiza a convolução da imagem segundo filtro Altas Genérico
    def Convolution_PassaAlta(self):
        cv2.destroyAllWindows()
        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
        self.Convolution(kernel, 0)
        cv2.imshow('Laplace Filter Image', self.img)
    # Realiza a convolução da imagem segundo filtro laplaciano
    def Convolution_Laplaciano(self):
        cv2.destroyAllWindows()
        kernel = [[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]]
        self.Convolution(kernel, 1)
        cv2.imshow('Laplace Filter Image', self.img)

    # Realiza a convolução da imagem segundo filtro gaussiano
    def Convolution_Gaussiano(self):
        cv2.destroyAllWindows()
        kernel = [[0.0625, 0.125, 0.0625],
                  [0.125, 0.25, 0.125],
                  [0.0625, 0.125, 0.0625]]
        self.Convolution(kernel, 0)
        cv2.imshow('Gauss Filter Image', self.img)

    #Realiza a operação de convolução do pixel
    def Convolution(self, kernel, condition):
        height, width, channel = self.img.shape
        for x in range(1, height-1):
            for y in range(1, width-1):
                M_Img = self.Create_Matrix_Img(x, y)
                for i in range(0, len(kernel)):
                    for j in range(0, len(kernel[0])):
                        self.soma += kernel[i][j]*M_Img[i][j]
                if condition:
                    self.soma += 127
                if self.soma > 255:
                    self.soma = 255
                if self.soma < 0:
                    self.soma = 0
                self.img[x, y] = self.soma
                self.soma = 0
    #Realiza a criação da matriz 3x3 referente ao pixel que será convoluido
    def Create_Matrix_Img(self, x, y):
        aux = np.zeros((3,3), dtype=np.int)
        j = -1
        for i in range(0, 3):
            aux[0][i] = self.img[x-1, y+j, 0]
            j += 1
        j = -1
        for i in range(0, 3):
            aux[1][i] = self.img[x, y+j, 0]
            j += 1
        j = -1
        for i in range(0, 3):
            aux[2][i] = self.img[x+1, y+j, 0]
            j += 1
        return aux

    # Realiza o zoom in da imagem utilizando a função resize do cv2
    def ZoomIn_Image(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        scale = 2
        #função do cv2 que com a imagem e a escala de aumento para o eixo x e y realiza a interpolação da matriz
        resized_image = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        self.img = resized_image
        cv2.imshow('Zoom_x2 Image', self.img)


    # Realiza o zoom out da imagem
    def ZoomOut_Image(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        scale_x = input('Digite a escala de diminuição do eixo x da imagem:')
        scale_y = input('Digite a escala de diminuição do eixo y da imagem:')
        # função do cv2 que com a imagem e a escala de alteração para o eixo x e y realiza a interpolação da matriz
        resized_image = cv2.resize(self.img, None, fx=1/int(scale_x), fy=1/int(scale_y), interpolation=cv2.INTER_LINEAR)
        self.img = resized_image
        cv2.imshow('Zoom_x2 Image', self.img)

    #Realiza a equalização da imagem
    def Equalize_Image(self):
        cv2.destroyAllWindows()
        plot.close()
        height, width, channel = self.img.shape
        hist = self.Histogram_Calc()
        hist_cum = np.array([sum(hist[:i+1]) for i in range(len(hist))]) #calculo do histograma cumulativo
        transfer_func = np.uint8(255 * hist_cum)
        Zero_Img = np.zeros_like(self.img)   #Retorna um array de zeros com a mesma forma e tipo que o array do parâmetro
        for i in range(0, height):
            for j in range(0, width):
                Zero_Img[i, j] = transfer_func[self.img[i, j]]    #Valores da imagem equalizada
        self.img = Zero_Img
        hist_equalized = self.Histogram_Calc()
        cv2.imshow('Equalized Image', self.img)
        # Normalização e plot do histograma equalizado
        plot.plot(((np.array(hist_equalized) - min(hist_equalized)) / (max(hist_equalized) - min(hist_equalized))) * 255)
        plot.ylabel('Equalized Histogram')
        plot.show()


    #Calcula o negativo da imagem
    def Negative_Calculation(self):
        cv2.destroyAllWindows()
        self.img = 255 - self.img
        cv2.imshow('Negative Image', self.img)


    # Ajusta o brilho da imagem de acordo com um valor informado
    def Adjust_Brightness(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        brightness = self.brightness.get()
        if int(brightness) >= -255 and int(brightness) <= 255:
            for i in range(0, height):
                for j in range(0, width):
                    self.Range_Test_Brightness(i, j, brightness)
                    if self.img[i, j, 0] != 0 and self.img[i, j, 0] != 255:
                        self.img[i, j, 0] += int(brightness)  # 0 -> canal azul
                    if self.img[i, j, 1] != 0 and self.img[i, j, 1] != 255:
                        self.img[i, j, 1] += int(brightness)  # 1 -> canal verde
                    if self.img[i, j, 2] != 0 and self.img[i, j, 2] != 255:
                        self.img[i, j, 2] += int(brightness)  # 2 -> canal vermelho
            cv2.imshow('Bright Image', self.img)

    #Teste de range para o brilho
    def Range_Test_Brightness(self, i, j, brightness):
        if self.img[i, j, 0] + int(brightness) < 0:
            self.img[i, j, 0] = 0
        if self.img[i, j, 0] + int(brightness) > 255:
            self.img[i, j, 0] = 255
        if self.img[i, j, 1] + int(brightness) < 0:
            self.img[i, j, 1] = 0
        if self.img[i, j, 1] + int(brightness) > 255:
            self.img[i, j, 1] = 255
        if self.img[i, j, 2] + int(brightness) < 0:
            self.img[i, j, 2] = 0
        if self.img[i, j, 2] + int(brightness) > 255:
            self.img[i, j, 2] = 255


    # Ajusta o contraste da imagem de acordo com um valor informado
    def Adjust_Contrast(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        contrast = self.contrast.get()
        if float(contrast) >= 0 and float(contrast) <= 255:
            for i in range(0, height):
                for j in range(0, width):
                    self.Range_Test_Contrast(i, j, contrast)
                    if self.img[i, j, 0] != 0 and self.img[i, j, 0] != 255:
                        self.img[i, j, 0] *= float(contrast)  # 0 -> canal azul
                    if self.img[i, j, 1] != 0 and self.img[i, j, 1] != 255:
                        self.img[i, j, 1] *= float(contrast)  # 1 -> canal verde
                    if self.img[i, j, 2] != 0 and self.img[i, j, 2] != 255:
                        self.img[i, j, 2] *= float(contrast)  # 2 -> canal vermelho
            cv2.imshow('Contrast Image', self.img)

    # Teste de range para o contraste
    def Range_Test_Contrast(self, i, j, contrast):
        if self.img[i, j, 0] * float(contrast) < 0:
            self.img[i, j, 0] = 0
        if self.img[i, j, 0] * float(contrast) > 255:
            self.img[i, j, 0] = 255
        if self.img[i, j, 1] * float(contrast) < 0:
            self.img[i, j, 1] = 0
        if self.img[i, j, 1] * float(contrast) > 255:
            self.img[i, j, 1] = 255
        if self.img[i, j, 2] * float(contrast) < 0:
            self.img[i, j, 2] = 0
        if self.img[i, j, 2] * float(contrast) > 255:
            self.img[i, j, 2] = 255

    #Mostra em um plot o histograma de uma imagem cinza
    def Plot_Histogram(self):
        plot.close()
        hist = self.Histogram_Calc()
        plot.plot(((np.array(hist) - min(hist)) / (max(hist) - min(hist)))*255)  #Normalização do histograma
        plot.ylabel('Normalized Histogram')
        plot.show()

    # Calcula o histograma de uma imagem cinza
    def Histogram_Calc(self):
        height, width, channel = self.img.shape
        n_hist = [0.0] * 256  # calculo de histograma normalizado
        for i in range(height):
            for j in range(width):
                n_hist[self.img[i, j, 0]] += 1
                #n_hist[self.img[i, j, 1]] += 1
                #n_hist[self.img[i, j, 2]] += 1
        hist = (np.array(n_hist) / (height * width))
        return hist

    # Realiza a rotação de 90 graus da imagem
    def Rotate_90_antih(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        Matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)  # Obtém a matriz de rotação para um determinado ângulo
                                                                          # e um centro determinado pela largura e altura da imagem
        cos = np.abs(Matrix[0, 0])                      #Valores de seno e coseno da matrix de rotação
        sin = np.abs(Matrix[0, 1])
        newWidth = int((height * sin) + (width * cos))   #Novas bordas da imagem
        newHeight = int((height * cos) + (width * sin))
        Matrix[0, 2] += (newWidth / 2) - (width / 2)       #Ajusta a translação da imagem para impedir que as bordas sejam cortadas
        Matrix[1, 2] += (newHeight / 2) - (height / 2)
        rotated_img = cv2.warpAffine(self.img, Matrix, (newWidth, newHeight))  # Aplica a matriz de rotação à matriz da imagem
                                                                              # para obter a imagem rotacionada
        self.img = rotated_img.copy()
        cv2.imshow('Rotate_counterclockwise', self.img)

    def Rotate_90_h(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        Matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, -1)  # Obtém a matriz de rotação para um determinado ângulo
                                                                          # e um centro determinado pela largura e altura da imagem
        cos = np.abs(Matrix[0, 0])  # Valores de seno e coseno da matrix de rotação
        sin = np.abs(Matrix[0, 1])
        newWidth = int((height * sin) + (width * cos))  # Novas bordas da imagem
        newHeight = int((height * cos) + (width * sin))
        Matrix[0, 2] += (newWidth / 2) - (width / 2)  # Ajusta a translação da imagem para impedir que as bordas sejam cortadas
        Matrix[1, 2] += (newHeight / 2) - (height / 2)
        rotated_img = cv2.warpAffine(self.img, Matrix, (newWidth, newHeight))  # Aplica a matriz de rotação à matriz da imagem
                                                                         # para obter a imagem rotacionada
        self.img = rotated_img.copy()
        cv2.imshow('Rotate_clockwise', self.img)

    #Verifica se é possível fazer a quantização com o número de tons informado
    def Verify_tones_number(self):
        tones_number = self.tones.get()
        if int(tones_number) >= 0 and int(tones_number) <= 255:
            self.quantization()

    # Realiza a quantização para um determinado número de tons
    def quantization(self):
        tones_number = self.tones.get()
        tones_factor = 255/int(tones_number)
        cv2.destroyAllWindows()
        self.img = np.uint8(self.img/tones_factor)*tones_factor  # Utiliza uma função para utilizar valores de 0 a 255
        cv2.imshow('Quantized Image', self.img)


    # Realiza a conversão para tons de cinza utilizando L = 0.299*R + 0.587*G + 0.114*B
    def convert_to_gray(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        for i in range(0, height):
            for j in range(0, width):
                pixel_blue = self.img[i, j, 0]   # 0 -> canal azul
                pixel_green = self.img[i, j, 1]  # 1 -> canal verde
                pixel_red = self.img[i, j, 2]    # 2 -> canal vermelho
                L = 0.299*pixel_red + 0.587*pixel_green + 0.114*pixel_blue
                self.img[i, j] = [L, L, L]  #altera o pixel atual para a tonalidade cinza
        cv2.imshow('Gray Image', self.img)


    # Realiza a rotação horizontal da imagem
    def rotation_horizontal(self):
        cv2.destroyAllWindows()
        rotated_img = cv2.flip(self.img, 1)
        self.img = rotated_img.copy()
        cv2.imshow('Rotate_h', self.img)


    # Realiza a rotação vertical da imagem
    def rotation_vertical(self):
        cv2.destroyAllWindows()
        height, width, channel = self.img.shape
        Matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 180,1)  # Obtém a matriz de rotação para um determinado ângulo
                                                                          # e um centro determinado pela largura e altura da imagem
        rotated_img = cv2.warpAffine(self.img, Matrix, (width, height))  # Aplica a matriz de rotação à matriz da imagem
                                                                         # para obter a imagem rotacionada
        self.img = rotated_img.copy()
        cv2.imshow('Rotate_v', self.img)

    #Utiliza função do opencv para mostrar a imagem modificada
    def show_current(self):
        cv2.destroyAllWindows()
        cv2.imshow('Show_current', self.img)

    # Utiliza função do opencv para mostrar a imagem sem alterações
    def copy_original(self):
        cv2.destroyAllWindows()
        self.img = img.copy()
        cv2.imshow('Copy_original', img)

    #Salva a imagem atual
    def save(self):
        cv2.imwrite(img_name[:-4]+'_modified.jpg', self.img)


# MAIN
img_name = input('Digite o nome da imagem desejada(com a extensão):')
img = cv2.imread(img_name)
img_copy = img.copy()


# Definição de alguns elementos da janela de interface
root = tk.Tk()
root.geometry('150x760+0+0')
root.configure(background = 'blue')
root.title('Trabalho 1 de FPI')
app = interface(img_copy, master=root)

app.mainloop()
import flet as ft
from flet import UserControl # new
import base64
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
aplicarFiltro = 0

#class Countdown(ft.UserControl):
class Countdown(UserControl):
    def __init__(self):
        super().__init__()

    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        while True:
            _, frame = cap.read()
            # frame = cv2.resize(frame,(400,400))
            
            if aplicarFiltro == 1:
                frame = self.montajeEsquina(frame)  #np.copy(frame)
            elif aplicarFiltro == 2:
                frame = self.montajeColor(frame)  #np.copy(frame)

            _, im_arr = cv2.imencode('.png', frame)
            im_b64 = base64.b64encode(im_arr)
            self.img.src_base64 = im_b64.decode("utf-8")
            self.update()

    def build(self):
        self.img = ft.Image(
            border_radius=ft.border_radius.all(20)
        )
        return self.img

    def montajeEsquina(self, cam_frame):
        # Capture the video frame by frame 
        #_, cam_frame = cap.read() 
        #image = cv2.imread('titan_1_azul.png')  #  'titan_2_azul.png' evaCrop.png
        image = cv2.imread('img/ctree_bluescreen.jpg')  #  'titan_2_azul.png' evaCrop.png
        #print('Image type: ', type(image),
        #'Image Dimensions : ', image.shape)

        hc,wc,cc = cam_frame.shape # imagen de la camara
        h,w,c = image.shape  # imagen montaje 

        if hc < h:   # Comprobar tamanios de las imagenes
            escala = 0.350
        else:
            escala = 0.75
        
        h_final = int(h*escala)
        w_final = int(w*escala)

        image_copy = np.copy(image)
        dsize = (w_final, h_final)

        # escalar imagen montaje
        image_copy = cv2.resize(image_copy, dsize)

        # Rango del filtrado    
        lower_blue = np.array([100, 0, 0])   
        upper_blue = np.array([255, 100, 120]) 

        # Crear mascara segun rango de colores
        mask = cv2.inRange(image_copy, lower_blue, upper_blue)
        #plt.imshow(mask, cmap='gray')

        # aplicar mascara a la imagen montaje
        masked_image = np.copy(image_copy)
        masked_image[mask != 0] = [0, 0, 0]

        # dimensiones de la mascara
        hp,wp,cp = masked_image.shape

        # Seleccionar la zona de la imagen 
        # donde va el montaje
        crop = cam_frame[(hc-hp):hc, 0:wp]

        # Separo en el CROP la zona donde va el montaje
        crop[mask == 0] = [0, 0, 0]

        # Union del crop y el montaje
        merged = crop + masked_image
       
        # Agregar a la imagen orignal el crop modificado
        cam_frame[(hc-hp):hc, 0:wp] =  merged
            
        return cam_frame
        
    def montajeColor(self, cam_frame):
        # Cargar la imagen
        #image = cv2.imread('queen3.png')
        image = cv2.imread('img/ctree_bluescreen.jpg')

        # rangos de color
        lower_col = np.array([0, 255, 50])   
        upper_col = np.array([0, 255, 100]) 

        mask = cv2.inRange(image, lower_col, upper_col)

        # Aplicar la máscara a la imagen usando la operación AND
        roi = cv2.bitwise_and(image, image, mask=mask)

        # Encontrar los contornos de la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar el rectángulo delimitador (bounding box) más pequeño alrededor de la región de interés
        x, y, w, h = cv2.boundingRect(contours[0])

        # Recortar la región de interés basada en el bounding box
        roi_cropped = roi[y:y+h, x:x+w]

        # Cargar la imagen que se va a insertar (la imagen que queremos superponer)
        overlay = np.copy(cam_frame)
        overlay = cv2.resize(overlay,roi_cropped.shape[1::-1])

        image[y:y+h, x:x+w] = overlay

        return image

def button_clicked(e):
    global aplicarFiltro
    aplicarFiltro = 1

def button_clicked2(e):
    global aplicarFiltro
    aplicarFiltro = 2

def button_clicked3(e):
    global aplicarFiltro
    aplicarFiltro = 0

section = ft.Container(
    margin=ft.margin.only(bottom=80),
    content=ft.Row([
        ft.Card(
            elevation=30,
            content=ft.Container(
                bgcolor=ft.colors.WHITE24,
                padding=25,
                border_radius = ft.border_radius.all(20),
                content=ft.Column([
                    Countdown(),
                    ft.Text("OPENCV con FLET", size=20, weight="bold",color=ft.colors.BLUE),
                    ft.Row([
                        ft.ElevatedButton("Aplicar Montaje", on_click=button_clicked),
                        ft.ElevatedButton("Aplicar Montaje Por Color", on_click=button_clicked2),
                        ft.ElevatedButton("No Aplicar", on_click=button_clicked3)], 
                        alignment=ft.MainAxisAlignment.CENTER)
                ]
                ),
            )
        )
      
    ],
        alignment=ft.MainAxisAlignment.CENTER
    )
)

def main(page: ft.Page):
    page.padding = 50
    page.window.left = page.window.left+100
    page.window.width = 1200        # window's width is 200 px
    page.window.height = 800       # window's height is 200 px
    page.theme_mode = ft.ThemeMode.LIGHT
    page.add(
        section
    )

if __name__ == '__main__':
    ft.app(target=main)  #, view=ft.AppView.WEB_BROWSER
    cap.release()
    cv2.destroyAllWindows()

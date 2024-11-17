import cv2

################################################################################################################################

# Leer imagen
im = cv2.imread('img/tower.jpg')
# Dimensiones de la Imagen
h,w,c = im.shape
print("Dimensiones de la imagen - Alto: {}, Ancho: {}, Canales: {}".format(h, w, c))
# Numero total de Pı́xeles
tot_pix = im.size
print("Numero total pixeles: {}".format(tot_pix))
# Obtener el tipo de datos de la imagen
image_dtype = im.dtype
# Imprimir tipo de datos de la imagen:
print("Tipo de datos de la imagen: {}".format(image_dtype))

################################################################

#Crear una versión en escala de grises
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

################################################################

# Mostrar imagen Original
cv2.imshow("Imagen Orginal", im)
# Mostrar Imagen en Escalas de Grises
cv2.imshow("Imagen en Escala de Grises", gray)
# Establecer espera
cv2.waitKey(0)
# Para destruir todas las ventanas creadas
cv2.destroyAllWindows()

################################################################

# Nuevo Tamaño
dsize = (int(w*0.250), int(h*0.250))
# escalar imagen
gray_rz = cv2.resize(gray, dsize)
im_rz = cv2.resize(im, dsize)
cv2.imshow("Imagen en Escala de Grises Escalada", gray_rz )
cv2.imshow("Imagen Orginal Escalada", im_rz)
cv2.waitKey(0)
# Para destruir todas las ventanas creadas
cv2.destroyAllWindows()

################################################################

# guardar imagen
cv2.imwrite("src/lab1/out/gray.png",gray)
cv2.imwrite("src/lab1/out/tower2.png",im_rz)

################################################################
################################################################
################################################################
################################################################

# Obtener colores de un pı́xel
(b, g, r) = im[6, 40]
# Imprimir los valores:
print("Pixel (6,40) - Rojo: {}, Verde: {}, Azul: {}".format(r, g, b))

################################################################

# obtener información de un canal
b = im[6, 40, 0]
print("Pixel (6,40) - Azul: {}".format(b))

################################################################

# Colocar un pı́xel en rojo
im_rz[6, 40] = (0, 0, 255)
# Obtener el valor del pı́xel (x=40, y=6) después de su modificación
(b, g, r) = im_rz[6, 40]
print("Píxel (6,40) - Rojo: {}, Verde: {}, Azul: {}".format(r, g, b))
cv2.imshow("Modificación de Píxel", im_rz)
cv2.waitKey(0)
# Para destruir todas las ventanas creadas
cv2.destroyAllWindows()

################################################################

# En este ejemplo utilizamos la esquina sup. izq de la imagen
roi_izq_sup = im_rz[0:50, 0:50]
# Mostramos esta región de interés (ROI):
cv2.imshow("Esquina Superior Izquierda Original", roi_izq_sup)
cv2.waitKey(0)

################################################################

# Se copia este parche a otra zona de la imagen
im_rz[20:70, 20:70] = roi_izq_sup
# Mostrar imagen modificada
cv2.imshow("Imagen modificada", im_rz)
cv2.waitKey(0)

################################################################

# Cambiar el color de una zona a azul
im_rz[0:50, 0:50] = (255, 0, 0)
# Mostrar imagen modificada
cv2.imshow("Imagen Modificada", im_rz)
cv2.waitKey(0)
cv2.destroyAllWindows()
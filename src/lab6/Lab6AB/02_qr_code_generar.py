import pyqrcode
import png
from PIL import Image

link= input("Ingrese la información que desea incluir en el QR: ")

# Generar Código QR
qr_code =pyqrcode.create(link)
qr_code.png("QRCode.png", scale=5)

# Mostrar Codigo  
qr_code.show()

# Mostrar la Imagen luego de guardarla
#Image.open("QRCode.png").show()


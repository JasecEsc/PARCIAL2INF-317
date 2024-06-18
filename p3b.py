import scipy.sparse as sp
import cv2
from google.colab import drive
import os
import matplotlib.pyplot as plt
# Montar Google Drive
drive.mount("/content/drive")
# Ruta de tu archivo CSV en Google Drive
os.chdir("/content/drive/My Drive/data")
imagen1=cv2.imread("imagen1.png")
imagen2=cv2.imread("imagen2.png")
resta = cv2.subtract(imagen1,imagen2)
plt.imshow(resta)
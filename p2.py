import scipy.sparse as sp
import cv2
from google.colab import drive
import os
# Montar Google Drive
drive.mount("/content/drive")
# Ruta de tu archivo CSV en Google Drive
os.chdir("/content/drive/My Drive/data")
imagen1=cv2.imread("imagen1.png")
imagen2=cv2.imread("imagen2.png")
m_sparce1=sp.coo_matrix(imagen1[:,:,1])
m_sparce2=sp.coo_matrix(imagen2[:,:,1])
print(m_sparce1)
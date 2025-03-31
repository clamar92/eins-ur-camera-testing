import cv2
import matplotlib.pyplot as plt

img_width = 640  # Larghezza dell'immagine
img_height = 480  # Altezza dell'immagine


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

# Cattura una singola immagine
ret, frame = cap.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
plt.show()
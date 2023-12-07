import cv2
import numpy as np
import os

# Membuat recognizer dan membaca model yang telah dilatih
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Membaca model yang telah dilatih
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Inisialisasi variabel
names = ['None', 'Yanto', 'Rio Jawir', 'Ilza', 'Z', 'W']  # Nama yang terkait dengan ID

# Inisialisasi dan mulai capture video secara real-time
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set lebar video
cam.set(4, 480)  # Set tinggi video

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Fungsi untuk membaca dan menampilkan frame
def read_and_display():
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            id, _ = recognizer.predict(gray[y:y+h, x:x+w])

            if 0 <= id < len(names):  # Ensure the id is within the valid range
                name = names[id]
            else:
                name = "unknown"

            cv2.putText(img, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

# Membuat thread untuk fungsi read_and_display
read_and_display()

# Menutup program dan membersihkan
print("\n [INFO] Menutup program dan membersihkan")
cam.release()
cv2.destroyAllWindows()

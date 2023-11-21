import cv2
import numpy as np
import os
import threading

# Membuat recognizer dan membaca model yang telah dilatih
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Membaca model yang telah dilatih
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Inisialisasi variabel
id = 0
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

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

# Membuat thread untuk fungsi read_and_display
thread_display = threading.Thread(target=read_and_display)

# Memulai thread
thread_display.start()

# Menunggu thread_display selesai
thread_display.join()

# Menutup program dan membersihkan
print("\n [INFO] Menutup program dan membersihkan")
cam.release()
cv2.destroyAllWindows()

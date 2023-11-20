import cv2
import numpy as np
from PIL import Image
import os

# Path untuk database gambar wajah
path = 'D:\COOLYEAH CERIYA CERIYA\SEMESTER 5\SISTEM PARALEL TERDISTRIBUSI\FaceDetector Project\dataset'

# Membuat objek recognizer dan detektor wajah
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Fungsi untuk mendapatkan data gambar dan label
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Mengonversi gambar ke grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        # Mendapatkan ID dari nama file gambar
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Mendeteksi wajah pada gambar
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            # Menyimpan bagian wajah sebagai sampel
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n [INFO] Melatih wajah. Ini akan memakan waktu beberapa detik. Tunggu ...")

# Mendapatkan data gambar dan label
faces, ids = getImagesAndLabels(path)

# Melatih recognizer dengan data gambar dan label
recognizer.train(faces, np.array(ids))

# Menyimpan model ke dalam file trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Menampilkan jumlah wajah yang dilatih dan mengakhiri program
print("\n [INFO] {0} wajah dilatih. Menutup Program".format(len(np.unique(ids))))

import cv2
import numpy as np
from PIL import Image
import os
import threading

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

# Fungsi untuk melatih recognizer
def trainRecognizer(start, end, faces, ids):
    print("\n [INFO] Melatih wajah dari dataset {0} hingga {1}. Tunggu ...".format(start, end))

    # Melatih recognizer dengan data gambar dan label
    recognizer.train(faces[start:end], np.array(ids[start:end]))

# Fungsi utama
def main():
    # Jumlah thread yang akan digunakan
    num_threads = 4

    # Membagi dataset menjadi bagian-bagian untuk setiap thread
    dataset_size = 50  # Update this to the actual size of your dataset
    batch_size = dataset_size // num_threads

    # Mendapatkan data gambar dan label
    faces, ids = getImagesAndLabels(path)

    # List untuk menyimpan objek thread
    threads = []

    # Mulai thread pelatihan
    for i in range(num_threads):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size if i != num_threads - 1 else dataset_size

        # Buat thread untuk melatih recognizer dengan bagian dataset tertentu
        thread = threading.Thread(target=trainRecognizer, args=(start_index, end_index, faces, ids.copy()))

        # Menambahkan thread ke dalam list
        threads.append(thread)

        # Mulai thread
        thread.start()

    # Tunggu semua thread selesai sebelum melanjutkan program utama
    for thread in threads:
        thread.join()

    # Menyimpan model ke dalam file trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')

    # Menampilkan jumlah wajah yang dilatih
    print("\n [INFO] {0} wajah dilatih.".format(len(np.unique(ids))))

    # Menutup program
    print("Menutup Program")

if __name__ == "__main__":
    main()

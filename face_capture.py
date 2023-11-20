import cv2
import os

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set lebar video
cam.set(4, 480)  # Set tinggi video

# Menggunakan Cascade Classifier untuk mendeteksi wajah
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Meminta pengguna untuk memasukkan ID wajah
face_id = input('\n Masukkan ID pengguna dan tekan <return> ==>  ')

print("\n [INFO] Memulai pengambilan wajah. Arahkan kamera dan tunggu ...")
count = 0  # Menghitung jumlah sampel wajah

# Loop untuk mengambil sampel wajah
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Menampilkan kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Menyimpan gambar wajah yang terdeteksi
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Tekan 'ESC' untuk keluar dari video
        break
    elif count >= 30:  # Ambil 30 sampel wajah dan hentikan video
        break

# Membersihkan
print("\n [INFO] Menutup program dan membersihkan")
cam.release()
cv2.destroyAllWindows()

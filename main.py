
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================
# 1. MEMBACA GAMBAR
# =====================================

img = cv2.imread("Foto1.JPG")

if img is None:
    raise FileNotFoundError("Foto1.jpeg tidak ditemukan!")

# Mengubah BGR menjadi RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Membuat folder hasil_filter
os.makedirs("hasil_filter", exist_ok=True)


# =====================================
# 2. MEAN FILTER
# =====================================

mean3 = cv2.blur(rgb, (3, 3))
mean5 = cv2.blur(rgb, (5, 5))


# =====================================
# 3. GAUSSIAN FILTER
# =====================================

gauss3 = cv2.GaussianBlur(rgb, (3, 3), 0)
gauss5 = cv2.GaussianBlur(rgb, (5, 5), 0)


# =====================================
# 4. MEDIAN FILTER
# =====================================

median3 = cv2.medianBlur(rgb, 3)
median5 = cv2.medianBlur(rgb, 5)


# =====================================
# 5. SHARPENING
# =====================================

kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

sharp = cv2.filter2D(rgb, -1, kernel)


# =====================================
# 6. MENAMPILKAN HASIL FILTER
# =====================================

images = [
    rgb, mean3, mean5, gauss3,
    gauss5, median3, median5, sharp
]

titles = [
    "Citra Asli",
    "Mean Filter 3x3",
    "Mean Filter 5x5",
    "Gaussian Filter 3x3",
    "Gaussian Filter 5x5",
    "Median Filter 3x3",
    "Median Filter 5x5",
    "Sharpening"
]

fig, ax = plt.subplots(2, 4, figsize=(16, 8))

for a, im, title in zip(ax.flatten(), images, titles):
    a.imshow(im)
    a.set_title(title)
    a.axis("off")

plt.tight_layout()

# Menyimpan gambar perbandingan
plt.savefig(
    "hasil_filter/perbandingan_filter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================
# 7. MENYIMPAN HASIL FILTER
# =====================================

for im, title in zip(images, titles):
    filename = title.lower().replace(" ", "_").replace("x", "x")
    output = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        f"hasil_filter/{filename}.jpg",
        output
    )


# =====================================
# 8. MENAMPILKAN NILAI INTENSITAS
# =====================================

print("\n===== NILAI INTENSITAS CITRA =====")

# Mengambil koordinat tengah gambar
tinggi, lebar = rgb.shape[:2]
y = tinggi // 2
x = lebar // 2

print(f"Ukuran gambar: {lebar} x {tinggi} piksel")
print(f"Koordinat piksel: ({x}, {y})")

for im, title in zip(images, titles):

    # Mengubah gambar RGB menjadi grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Mengambil nilai intensitas pada titik tengah
    nilai = gray[y, x]

    # Menghitung rata-rata intensitas seluruh gambar
    rata_rata = np.mean(gray)

    print(f"\n{title}")
    print(f"Intensitas piksel tengah: {nilai}")
    print(f"Rata-rata intensitas: {rata_rata:.2f}")

print("\nSemua proses selesai!")
print("Hasil gambar tersimpan di folder hasil_filter.")
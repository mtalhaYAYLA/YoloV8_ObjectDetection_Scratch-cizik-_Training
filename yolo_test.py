import cv2
from ultralytics import YOLO

# def enhance_contrast(frame):
#     # 1. Gri tonlamaya çevir
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 2. CLAHE uygula
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_gray = clahe.apply(gray)

#     # 3. Gri resmi tekrar BGR formatına çevir (YOLO modeli renkli bekler)
#     enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

#     return enhanced_bgr


def adjust_brightness_contrast(image, alpha=1.5, beta=20):
    """
    Görüntünün kontrast ve parlaklığını artırır.
    
    alpha: kontrast çarpanı (1.0 = orijinal, >1 = daha kontrastlı)
    beta: parlaklık değeri (0 = orijinal, >0 = daha parlak)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_contrast(frame, alpha=1.5, beta=20):
    # 1. Griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE uygula (lokal kontrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 3. Tekrar renkliye çevir (YOLO renkli ister)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # 4. Genel kontrast ve parlaklık ayarı
    final = adjust_brightness_contrast(enhanced_bgr, alpha=alpha, beta=beta)

    return final


# === Ayarlar ===
model_path = r"C:\Users\aisoft\yaylabey\runs\detect\train8\weights\best.pt"  # Model yolu
video_path = r"C:\Users\aisoft\yaylabey\video\Video_20250527111527326.avi"  # Video yolu
output_path = r"C:\Users\aisoft\yaylabey\output_detected.mp4"  # Kaydedilecek video yolu (isteğe bağlı)
window_width, window_height = 800, 600  # Görüntü boyutu
save_output = True  # Sonucu kaydetmek ister misiniz?

# === Model ve Video Açılışı ===
model = YOLO(model_path)  # YOLOv8 modelini belirtilen yoldan yükler
cap = cv2.VideoCapture(video_path)  # Video dosyasını açar

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

# Video kaydı ayarları
out = None
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec ayarı (mp4 için)
    out = cv2.VideoWriter(
        output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
        (window_width, window_height)
    )  # Sonuç videosunu kaydetmek için VideoWriter nesnesi oluşturur

while True:
    ret, frame = cap.read()  # Videodan bir kare okur
    if not ret:
        print("Video bitti veya açılamadı.")
        break

    # YOLOv8 ile tahmin yap
    # results = model(frame)  # Kare üzerinde nesne tespiti yapar
    #results = model.predict(source=frame, conf=0.33, save=False, verbose=False)  # Kare üzerinde nesne tespiti yapar
    """
    burda yeni olarak CLAHE + Kontrast Kodları: olucak
    """
    # Görüntüyü iyileştir (CLAHE + Kontrast)
    enhanced_frame = enhance_contrast(frame)

    # YOLOv8 ile tahmin yap
    results = model.predict(source=enhanced_frame, conf=0.33, save=False, verbose=False)

    annotated_frame = results[0].plot()  # Tespit edilen nesneleri kare üzerine çizer

    # annotated_frame None veya boş ise atla
    if annotated_frame is None or annotated_frame.size == 0:
        print("Geçersiz görüntü geldi, atlanıyor.")
        continue

    # RGBA ise BGR'ye çevir (OpenCV için)
    if annotated_frame.shape[2] == 4:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)

    # Yeniden boyutlandır
    resized_frame = cv2.resize(annotated_frame, (window_width, window_height))  # Görüntüyü istenen boyuta getirir

    # Ekranda göster
    cv2.imshow('YOLOv8 Detection', resized_frame)  # Sonuçları ekranda gösterir

    # Kaydet
    if save_output and out is not None:
        out.write(resized_frame)  # Sonuç videosuna kareyi ekler

    # 'q' veya ESC tuşu ile çıkış
    key = cv2.waitKey(100) & 0xFF  # 100 ms bekler, video yavaşlar
    if key == ord('q') or key == 27:
        print("Çıkış yapıldı.")
        break

cap.release()  # Video dosyasını kapatır
if save_output and out is not None:
    out.release()  # Video kaydını kapatır
cv2.destroyAllWindows()  # Açık tüm OpenCV pencerelerini kapatır

# ============================
# FONKSİYONLARIN VE KODUN AÇIKLAMASI
# ============================
# model = YOLO(model_path)
#   -> Belirtilen yoldaki YOLOv8 modelini yükler. Nesne tespiti için kullanılır.
#
# cap = cv2.VideoCapture(video_path)
#   -> Video dosyasını kare kare okumak için açar.
#
# cap.isOpened()
#   -> Video dosyasının başarıyla açılıp açılmadığını kontrol eder.
#
# cv2.VideoWriter_fourcc(*'mp4v')
#   -> Video kaydı için kullanılacak codec'i belirler (mp4 formatı için).
#
# cv2.VideoWriter(...)
#   -> Sonuçları kaydetmek için video dosyası oluşturur.
#
# cap.read()
#   -> Videodan bir kare okur. ret: okuma başarılıysa True, frame: okunan görüntü.
#
# model(frame)
#   -> Verilen kare üzerinde nesne tespiti yapar, sonuçları döndürür.
#
# results[0].plot()
#   -> Tespit edilen nesneleri (bounding box, sınıf adı vs.) kare üzerine çizer.
#
# annotated_frame.shape[2] == 4
#   -> Görüntüde alfa kanalı (RGBA) olup olmadığını kontrol eder.
#
# cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)
#   -> RGBA görüntüyü BGR formatına çevirir (OpenCV uyumluluğu için).
#
# cv2.resize(annotated_frame, (window_width, window_height))
#   -> Görüntüyü istenen boyuta yeniden boyutlandırır.
#
# cv2.imshow('YOLOv8 Detection', resized_frame)
#   -> Sonuçları ekranda gösterir.
#
# out.write(resized_frame)
#   -> Sonuç videosuna işlenmiş kareyi ekler.
#
# cv2.waitKey(1)
#   -> Klavyeden tuş basımı bekler. 'q' veya ESC tuşuna basılırsa döngüden çıkar.
#
# cap.release()
#   -> Video dosyasını kapatır.
#
# out.release()
#   -> Video kaydını kapatır.
#
# cv2.destroyAllWindows()
#   -> Açık tüm OpenCV pencerelerini kapatır.
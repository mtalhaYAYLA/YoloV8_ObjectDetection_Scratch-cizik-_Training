import cv2
import numpy as np
from ultralytics import YOLO
import time

# =============================
# AYARLANABİLİR GÖRÜNTÜ İYİLEŞTİRME PARAMETRELERİ
# =============================
# CLIP_LIMIT = 3.0         # CLAHE kontrast artırma gücü (2.0 - 4.0 önerilir)
# TILE_GRID_SIZE = (4, 4)  # CLAHE bölge büyüklüğü (küçük değerler ince detayları vurgular)
# ALPHA = 1.8              # Kontrast çarpanı (1.5 - 2.0 arası önerilir)
# BETA = 15                # Parlaklık artışı (10 - 30 arası deneyebilirsin)
# GAMMA = 1.2              # Gamma düzeltmesi (1.0 - 1.5 arası)
"""
| Durum                                          | Ayar Tavsiyesi                                 |
| ---------------------------------------------- | ---------------------------------------------- |
| Bazı çizikler hâlâ çok soluk                   | `GAMMA → 1.3` veya `CLIP_LIMIT → 3.5` yap      |
| Arka plan çok parlıyorsa, detaylar bozuluyorsa | `BETA → 10` yap                                |
| Çizikler fazla parlıyorsa, patlama oluyorsa    | `ALPHA → 1.6`, `GAMMA → 1.1` olarak düşür      |
| Tespit edilen kutu sayısı hâlâ azsa            | `conf=0.25` ile tahmin yap, daha fazlası gelir |
"""

CLIP_LIMIT = 3.0         # CLAHE kontrast artırma gücü (2.0 - 4.0 önerilir)
TILE_GRID_SIZE = (4, 4)  # CLAHE bölge büyüklüğü (küçük değerler ince detayları vurgular)
ALPHA = 1.8              # Kontrast çarpanı (1.5 - 2.0 arası önerilir)
BETA = 8                # Parlaklık artışı (10 - 30 arası deneyebilirsin)
GAMMA = 1.2              # Gamma düzeltmesi (1.0 - 1.5 arası)


# =============================
# YOLO MODEL ve VİDEO DOSYALARI
# =============================
model_path = r"C:\Users\aisoft\yaylabey\runs\detect\train8\weights\best.pt"
video_path = r"C:\Users\aisoft\yaylabey\video\Video_20250527111527326.avi"
output_path = r"C:\Users\aisoft\yaylabey\output_detected.mp4"
window_width, window_height = 800, 600
save_output = True

# =============================
# İYİLEŞTİRME FONKSİYONLARI
# =============================

def adjust_brightness_contrast(image, alpha=1.5, beta=20):
    """Genel kontrast ve parlaklık artırımı yapar"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def adjust_gamma(image, gamma=1.2):
    """Gamma düzeltmesi yapar (açık detayları vurgular)"""
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)])).astype("uint8")
    return cv2.LUT(image, table)
#enhanced = enhance_contrast(image) şeklinde data seti elden geçir ve tekrtar eğit  
def enhance_contrast(frame, clip_limit=2.0, tile_grid_size=(8, 8), alpha=1.5, beta=20, gamma=1.2):
    """CLAHE + Kontrast-Parlaklık + Gamma uygulayarak görüntüyü iyileştirir"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Griye çevir
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_gray = clahe.apply(gray)
    bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    contrast_applied = adjust_brightness_contrast(bgr, alpha=alpha, beta=beta)
    final = adjust_gamma(contrast_applied, gamma=gamma)
    return final

# =============================
# MODEL ve VİDEO AÇILIŞI
# =============================
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

# Video kaydı için ayar
out = None
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (window_width, window_height))

# FPS hesaplaması için başlangıç zamanı
prev_time = time.time()

# =============================
# ANA DÖNGÜ
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya kare okunamadı.")
        break

    # Görüntüyü çizik tespiti için iyileştir
    enhanced_frame = enhance_contrast(
        frame,
        clip_limit=CLIP_LIMIT,
        tile_grid_size=TILE_GRID_SIZE,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA
    )

    # YOLO ile tahmin yap
    results = model.predict(source=enhanced_frame, conf=0.20, save=False, verbose=False)

    # Tahmin sonuçlarını çiz
    annotated_frame = results[0].plot()

    if annotated_frame is None or annotated_frame.size == 0:
        print("Boş kare alındı, atlanıyor.")
        continue

    # RGBA ise BGR'ye çevir
    if annotated_frame.shape[2] == 4:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)

    # Ekrana uygun boyuta getir
    resized_frame = cv2.resize(annotated_frame, (window_width, window_height))

    # FPS hesapla ve ekle
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tespit edilen nesne sayısı
    num_detections = len(results[0].boxes)
    cv2.putText(resized_frame, f"Tespit: {num_detections}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Ekranda göster
    cv2.imshow("YOLOv8 Çizik Tespiti", resized_frame)

    # Videoya kaydet
    if save_output and out is not None:
        out.write(resized_frame)

    # 'q' veya ESC ile çık
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:
        break

# Kaynakları kapat
cap.release()
if save_output and out is not None:
    out.release()
cv2.destroyAllWindows()

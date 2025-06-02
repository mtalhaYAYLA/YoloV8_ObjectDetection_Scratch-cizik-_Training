import cv2
from ultralytics import YOLO
from hik_camera import HikCamera  # kendi sınıfın

def main():
    # Eğitilmiş YOLOv8 modelini yükle
    model = YOLO(r"C:/Users/aisoft/yaylabey/runs/detect/train7/weights/best.pt")

    # HikCamera'yı başlat (ilk bulunan IP ile bağlanır)
    cam = HikCamera.get_cam()

    # Kamera açıldığında otomatik kapanması için with kullanılır
    with cam:
        while True:
            # Kameradan görüntü al (numpy.ndarray, RGB format)
            frame = cam.robust_get_frame()

            # YOLOv8 ile tahmin yap
            results = model.predict(source=frame, conf=0.25, save=False, verbose=False)

            # Tahmin sonuçlarını çiz (bounding box, class adı vs.)
            result_img = results[0].plot()

            # Görüntüyü %50 oranında küçült
            display_img = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)

            # Ekranda göster
            cv2.imshow("YOLOv8 HikCamera Live Detection", display_img)

            # 'q' tuşuna basınca çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Pencereyi kapat
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


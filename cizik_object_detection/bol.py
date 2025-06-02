import os
import shutil
import random

# Ana klasör yolu (burayı kendi sistemine göre düzenle)
base_dir = r"C:\Users\aisoft\yaylabey\cizik_object_detection"

# Kaynak klasörler
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Hedef klasörler (train ve valid içindeki images/labels alt klasörleri)
train_images = os.path.join(base_dir, "train", "images")
train_labels = os.path.join(base_dir, "train", "labels")
val_images = os.path.join(base_dir, "valid", "images")
val_labels = os.path.join(base_dir, "valid", "labels")

# Eğer varsa, bu hedef klasörleri komple sil ve sıfırdan oluştur
for folder in [train_images, train_labels, val_images, val_labels]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Görüntü dosyalarını listele ve karıştır
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# %80 eğitim, %20 doğrulama bölmesi
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Belirli listeye göre kopyalama fonksiyonu
def copy_files(file_list, dest_img_dir, dest_lbl_dir):
    for img_file in file_list:
        # Görüntü yolu
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(dest_img_dir, img_file)

        # Etiket yolu (.txt)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_lbl = os.path.join(labels_dir, label_file)
        dst_lbl = os.path.join(dest_lbl_dir, label_file)

        # Kopyala
        shutil.copy(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)
        else:
            print(f"UYARI: Etiket bulunamadı: {label_file}")

# Kopyalama işlemleri
copy_files(train_files, train_images, train_labels)
copy_files(val_files, val_images, val_labels)

print("✅ Tüm dosyalar başarıyla ayrıldı.")

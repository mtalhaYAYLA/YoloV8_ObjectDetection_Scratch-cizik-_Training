# train: path/to/train/images
# val: path/to/val/images

# nc: 3  # sınıf sayısı
# names: ['class1', 'class2', 'class3']



from ultralytics import YOLO

def train():
    model = YOLO('yolov8x.pt')
    model.train(
        data=r'C:\Users\aisoft\yaylabey\cizik_object_detection\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0
    )

if __name__ == '__main__':
    train()

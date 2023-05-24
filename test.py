import cv2
import numpy as np
import torch
from matplotlib import patches, pyplot as plt

def read_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # Преобразование изображения из BGR в RGB
    image /= 255.0  # Normalize
    return image

def read_bboxes_and_labels_from_txt(file_path):
    bboxes = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            label = data[0]
            bbox = [float(coord) for coord in data[1:]]  # Преобразование строковых координат в целые числа
            labels.append(float(label))
            bboxes.append(bbox)
    return labels, np.array(bboxes)

def mixup_augmentation(image1, labels1, bboxes1, image2, labels2, bboxes2, alpha=0.2):
    lam = 0.4

    mixed_image = lam * image1.astype(np.float32)  + (1 - lam) * image2.astype(np.float32)

    mixed_bboxes = []
    num_bboxes1 = bboxes1.shape[0]
    num_bboxes2 = bboxes2.shape[0]
    num_bboxes = max(num_bboxes1, num_bboxes2)

    for i in range(num_bboxes):
        if i < num_bboxes1 and i < num_bboxes2:
            mixed_bbox = lam * bboxes1[i] + (1 - lam) * bboxes2[i]
        elif i < num_bboxes1:
            mixed_bbox = bboxes1[i]
        else:
            mixed_bbox = bboxes2[i]

        mixed_bboxes.append(mixed_bbox)

    mixed_bboxes = np.array(mixed_bboxes)

    return mixed_image, mixed_bboxes


image1 = read_image('Codev.v3i.yolov8/test/images/0e4cdbd5-660a-4903-954d-51b40458043e_png_jpg.rf.76fad1025ee5b969623d11f72712ea00.jpg')
image2 = read_image('Codev.v3i.yolov8/test/images/1f6e825f-cabe-43aa-968a-6ea70e6ea9c1_png_jpg.rf.49770e1a2e15baf0d27871b1d76e46b5.jpg')
images=[image1,image2]
# Считываем классы и ограничивающие рамки для первого изображения
labels1, bboxes1 = read_bboxes_and_labels_from_txt('Codev.v3i.yolov8/test/labels/0e4cdbd5-660a-4903-954d-51b40458043e_png_jpg.rf.76fad1025ee5b969623d11f72712ea00.txt')

# Считываем классы и ограничивающие рамки для второго изображения
labels2, bboxes2 = read_bboxes_and_labels_from_txt('Codev.v3i.yolov8/test/labels/1f6e825f-cabe-43aa-968a-6ea70e6ea9c1_png_jpg.rf.49770e1a2e15baf0d27871b1d76e46b5.txt')
bbox=[labels1,labels2]
areas=[bboxes1,bboxes2]
# Применяем аугментацию Mixup
mixed_image, mixed_bboxes = mixup_augmentation(
    image1, labels1, bboxes1, image2, labels2, bboxes2, alpha=0.2
)
print(mixed_image)
cv2.imshow("",mixed_image)
cv2.waitKey(5000)

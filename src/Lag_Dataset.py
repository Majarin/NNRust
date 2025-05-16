import numpy as np
import cv2
import os
import random

# Konfigurasjon
img_size = 28
num_per_shape = 1000
shapes = {
    0: "circle",
    1: "triangle",
    2: "square",
    3: "pentagon",
    4: "hexagon",
}
num_classes = len(shapes)

def draw_shape(shape_id):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    radius = 10

    if shape_id == 0:  # Circle
        cv2.circle(img, center, radius, 255, -1)

    elif shape_id == 1:  # Triangle
        pts = np.array([
            [center[0], center[1] - radius],
            [center[0] - radius, center[1] + radius],
            [center[0] + radius, center[1] + radius],
        ], np.int32)
        cv2.fillPoly(img, [pts], 255)

    elif shape_id == 2:  # Square
        cv2.rectangle(img,
                      (center[0] - radius, center[1] - radius),
                      (center[0] + radius, center[1] + radius),
                      255, -1)

    elif shape_id == 3:  # Pentagon
        pts = regular_polygon(center, radius, 5)
        cv2.fillPoly(img, [pts], 255)

    elif shape_id == 4:  # Hexagon
        pts = regular_polygon(center, radius, 6)
        cv2.fillPoly(img, [pts], 255)

    return img

def regular_polygon(center, radius, sides):
    angle = 2 * np.pi / sides
    points = [
        (
            int(center[0] + radius * np.cos(i * angle - np.pi / 2)),
            int(center[1] + radius * np.sin(i * angle - np.pi / 2))
        )
        for i in range(sides)
    ]
    return np.array(points, np.int32)

# Lagre bilder og etiketter
images = []
labels = []

for shape_id in range(num_classes):
    for _ in range(num_per_shape):
        img = draw_shape(shape_id)

        # Valgfritt: Legg til litt tilfeldig rotasjon eller støy
        if random.random() < 0.3:
            angle = random.randint(-30, 30)
            rot_mat = cv2.getRotationMatrix2D((img_size // 2, img_size // 2), angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (img_size, img_size), borderValue=0)

        images.append(img.astype(np.float32) / 255.0)  # normaliser til [0,1]
        labels.append(shape_id)

# Konverter til numpy-arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.uint8)

print("Shape på datasettet:", images.shape, labels.shape)

# Lagre som .npz-fil
np.savez("shape_dataset.npz", images=images, labels=labels)
print("Datasett lagret som shape_dataset.npz")

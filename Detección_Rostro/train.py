#Para entrenar eimport cv2

import cv2
import os
import numpy as np

data_path = 'dataset'
people = os.listdir(data_path)
labels = []
faces = []
label_map = {}

for label, person in enumerate(people):
    label_map[label] = person
    person_path = os.path.join(data_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save('face_model.yml')

# Guardar etiquetas
with open("labels.txt", "w") as f:
    for label, name in label_map.items():
        f.write(f"{label}:{name}\n")

print("✅ Modelo entrenado y guardado.")

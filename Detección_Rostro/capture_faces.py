#Para capturar las fotos de cada persona desde la webcam

import cv2
import os

# 🟩 1. Solicitar nombre
person_name = input("Nombre de la persona: ").strip()
save_path = f'dataset/{person_name}'

# 🟩 2. Crear carpeta si no existe
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"📁 Carpeta creada: {save_path}")
else:
    print(f"📁 Guardando en carpeta existente: {save_path}")

# 🟩 3. Inicializar captura de video con backend AVFoundation (recomendado en macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara. Verifica permisos en Preferencias del Sistema > Seguridad > Cámara.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
max_images = 40

print("🎥 Iniciando captura... Presiona ESC para salir antes de tiempo.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Redimensionar rostro (opcional, LBPH no lo necesita pero es útil)
        rostro = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        cv2.imwrite(f"{save_path}/{count}.jpg", rostro)
        count += 1

        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Captura de rostros', frame)

    # Terminar si presiona ESC o ya se capturaron 40 imágenes
    if cv2.waitKey(1) == 27 or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Captura finalizada. Total imágenes: {count}")

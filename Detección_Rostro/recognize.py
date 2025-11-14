#Para hacer el reconocimiento facial en tiempo real
import cv2

# Cargar modelo
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_model.yml')

# Cargar etiquetas
label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        label, name = line.strip().split(":")
        label_map[int(label)] = name

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(rostro)

        name = label_map.get(label, "Desconocido")
        text = f"{name} ({round(confidence, 2)})"
        color = (0, 255, 0) if confidence < 80 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Reconocimiento Facial', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

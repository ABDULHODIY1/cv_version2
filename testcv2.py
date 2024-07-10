import torch
import cv2

# YOLOv5 modelini yuklash (faqat 'person' klassini aniqlash uchun)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Video tasmadan kadrlar olish
cap = cv2.VideoCapture(0)  # Real time video uchun, 0 - asosiy kamerani bildiradi

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Model yordamida aniqlash
    results = model(frame)

    # Annotatsiyalarni olish
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # Aniqlangan odamlar sonini sanash
    people_count = 0
    for i in range(len(labels)):
        if labels[i] == 0:  # 0 - 'person' klassi
            people_count += 1

    # Annotatsiyalarni chizish
    results.render()

    # Kadrni ko'rsatish
    cv2.putText(frame, f'People Count: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

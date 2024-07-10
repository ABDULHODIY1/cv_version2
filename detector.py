import numpy as np
import cv2
import datetime
import requests
import serial

class HumanDetector:
    """
    Class For Human Detection
    """

    def __init__(self):
        """
        Initialize the Human Detection
        """
        self.url = "192.168.192.91"
        self.port = "8080"
        self.hog = cv2.HOGDescriptor()
        self.cap = cv2.VideoCapture(0)
        self.out = cv2.VideoWriter(
            'output.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            15.,
            (640, 480))
        self.people_count = 0

    def detect(self):
        """
        Detector for human
        :return:
        """
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def send_signal(self, value, method="serial", port_name='/dev/ttyUSB0'):
        """
        port orqali signal yuborish uchun funksiya
        :param value: True qiymat jonatish uchun parametr
        :param method: Method signal jonatish uchun parametr
        :param port_name: Serial Port Nomi kiritish kerak
        :return: None
        """
        if method == "network":
            try:
                response = requests.post(f"http://{self.url}:{self.port}/signal", json={"signal": value})
                if response.status_code == 200:
                    print("Signal sent via network.")
                else:
                    print(f"Failed to send signal via network: {response.status_code}")
            except Exception as e:
                print(f"Network error: {e}")

        elif method == "serial":
            try:
                ser = serial.Serial(port_name, 9600, timeout=1)
                ser.write(f"{value}\n".encode())
                ser.close()
                print("Signal sent via serial port.")
            except Exception as e:
                print(f"Serial error: {e}")

    def signal(self, hour, count):
        """
        Odamlar soni va soatni taxlil qiluvchi funksiya
        :param hour: Soat
        :param count: Odamlar soni detectori
        :return: Boolean
        """
        if 6 <= hour < 17 and count >= 8:
            self.send_signal(True)
            print(True)
            return True
        elif 17 <= hour < 18 and count >= 4:
            self.send_signal(True)
            print(True)
            return True
        elif hour >= 18 and count >= 1:
            self.send_signal(True)
            print(True)
            return True
        return False

    def start(self):
        """
        Start funksiyasi kamerani ishga tushurish uchun
        :return:
        """
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get current hour
            now = datetime.datetime.now()
            hour = now.hour

            # resizing for faster detection
            frame = cv2.resize(frame, (640, 480))

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect people
            boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            # Update people count
            self.people_count += len(boxes)

            # Check and send signal
            if self.signal(hour, self.people_count):
                print(f"Signal sent at {now.strftime('%Y-%m-%d %H:%M:%S')} with {self.people_count} people detected.")
                self.people_count = 0  # Reset the count after sending the signal

            # Draw bounding boxes
            for (xA, yA, xB, yB) in boxes:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Write the output video
            self.out.write(frame.astype('uint8'))

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

# Create an instance of the HumanDetector class
app = HumanDetector()
app.detect()
app.start()

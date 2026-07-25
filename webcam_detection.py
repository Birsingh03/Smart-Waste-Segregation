import cv2
import tensorflow as tf
import numpy as np
import time
import serial

# Load trained model
model = tf.keras.models.load_model("models/garbage_classifier.keras")

# ---------------- Arduino ----------------

try:
    arduino = serial.Serial("COM9", 9600)
    time.sleep(2)
    print("Arduino Connected!")
except Exception as e:
    print("Arduino Not Connected!")
    print(e)
    arduino = None

# ---------------- Class Names ----------------

class_names = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

# ---------------- Bin Mapping ----------------

BIN_MAPPING = {
    "battery": "General Waste Bin",
    "biological": "Organic Bin",
    "cardboard": "Dry Waste Bin",
    "clothes": "General Waste Bin",
    "glass": "Dry Waste Bin",
    "metal": "Dry Waste Bin",
    "paper": "Dry Waste Bin",
    "plastic": "Plastic Bin",
    "shoes": "General Waste Bin",
    "trash": "General Waste Bin"
}

# ---------------- Arduino Function ----------------

def send_to_arduino(predicted_class):

    if arduino is None:
        return

    if predicted_class == "plastic":
        arduino.write(b'P')
        print("Sent -> P")

    elif predicted_class in ["metal", "glass", "paper", "cardboard"]:
        arduino.write(b'D')
        print("Sent -> D")

    else:
        arduino.write(b'G') # battery, biological, clothes, shoes, trash
        print("Sent -> G")

# ---------------- Settings ----------------

last_detection_time = 0
cooldown = 7

CONFIDENCE_THRESHOLD = 90

predicted_class = ""
confidence = 0
bin_name = ""

object_processed = False

# ---------------- Webcam ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit")

# First frame for motion detection
ret, previous_frame = cap.read()

if not ret:
    print("Failed to read webcam.")
    cap.release()
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# ---------------- Main Loop ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------- Motion Detection ----------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(previous_gray, gray)

    motion = np.sum(diff)

    previous_gray = gray.copy()

    if object_processed and motion > 2500000:

        object_processed = False

        predicted_class = ""
        confidence = 0
        bin_name = ""

        print("\nReady for next object...\n")

    # ---------- Prediction ----------

    current_time = time.time()

    if (not object_processed) and (current_time - last_detection_time >= cooldown):

        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = np.max(prediction) * 100

        if confidence >= CONFIDENCE_THRESHOLD:

            bin_name = BIN_MAPPING[predicted_class]

            send_to_arduino(predicted_class)

            print("\n------------------------------")
            print("Detected :", predicted_class)
            print(f"Confidence : {confidence:.2f}%")
            print("Open :", bin_name)
            print("------------------------------")

            object_processed = True
            last_detection_time = current_time

    # ---------- Display ----------

    cv2.putText(
        frame,
        f"Object : {predicted_class}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Bin : {bin_name}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Smart Waste Segregation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- Cleanup ----------------

cap.release()
cv2.destroyAllWindows()

if arduino is not None:
    arduino.close()
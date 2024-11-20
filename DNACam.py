import cv2
import pandas as pd
import datetime
import random
from collections import Counter
from Bio.Seq import Seq
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import RPi.GPIO as GPIO  # For button integration

# Configure GPIO for the button
BUTTON_PIN = 17  # GPIO pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

CELL_TYPES = ["Normal", "Cancerous", "Bacteria"]
DANGER_LEVELS = {"Normal": "Low", "Cancerous": "High", "Bacteria": "Moderate"}
all_cell_types, all_danger_levels, cells_detected_over_time, timestamps = [], [], [], []
is_running = False

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

def detect_cells(image):
    processed_image = preprocess_image(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def assign_cell_type():
    cell_type = random.choice(CELL_TYPES)
    danger_level = DANGER_LEVELS[cell_type]
    return cell_type, danger_level

def dna_analysis(cell_type):
    dna_sequences = {
        "Normal": Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"),
        "Cancerous": Seq("ATGGCGGCTTAGTGAAGCCGCTGAAAGGGTGACCGATAG"),
        "Bacteria": Seq("ATGCCTGCGTACGGCTAGTCAGAGCTGAGCGATCCTG")
    }
    dna_seq = dna_sequences[cell_type]
    rev_complement = dna_seq.reverse_complement()
    return str(dna_seq), str(rev_complement)

def log_data(contours_count, cell_type, danger_level, dna_seq, rev_complement):
    data = {
        'timestamp': [datetime.datetime.now()],
        'cells_detected': [contours_count],
        'cell_type': [cell_type],
        'danger_level': [danger_level],
        'dna_sequence': [dna_seq],
        'reverse_complement': [rev_complement]
    }
    df = pd.DataFrame(data)
    df.to_csv('/home/pi/cell_detection_log.csv', mode='a', header=False, index=False)

def plot_data():
    if timestamps and cells_detected_over_time:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, cells_detected_over_time, marker='o', label="Cells Detected")
        plt.title("Cells Detected Over Time")
        plt.xlabel("Time")
        plt.ylabel("Number of Cells Detected")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv
    return None

# CSV Initialization
headers = ['timestamp', 'cells_detected', 'cell_type', 'danger_level', 'dna_sequence', 'reverse_complement']
pd.DataFrame(columns=headers).to_csv('/home/pi/cell_detection_log.csv', mode='w', header=True, index=False)

# Camera Initialization
cap = cv2.VideoCapture(0)

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # Button pressed
            is_running = not is_running
            print("Detection toggled:", "Running" if is_running else "Stopped")
        
        if is_running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            contours = detect_cells(frame)
            contours_count = len(contours)
            cell_type, danger_level = assign_cell_type()
            dna_seq, rev_complement = dna_analysis(cell_type)

            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            display_text = f"Cells: {contours_count} | Type: {cell_type} | Danger: {danger_level}"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            log_data(contours_count, cell_type, danger_level, dna_seq, rev_complement)

            all_cell_types.append(cell_type)
            all_danger_levels.append(danger_level)
            cells_detected_over_time.append(contours_count)
            timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))

            cv2.imshow('Cell Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

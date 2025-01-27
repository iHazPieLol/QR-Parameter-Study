from qreader import QReader
from cv2 import imread
from pyzbar.pyzbar import decode
import pandas as pd
import cv2
import threading

# Read the CSV file containing image paths and parameters
csv_path = r"C:\Users\haroldj\OneDrive - scion\Documents\General\QR Code Research\Parameter Study\Virtual Parameter Study\rendered_spheres_mm.csv"  # Update this to your CSV file path
df = pd.read_csv(csv_path)

# Set the correct message expected in the QR codes
correct_message = "https://www.example.com"

# Initialize the QR code readers
qreader_reader = QReader()
qcd = cv2.QRCodeDetector()  # OpenCV's QRCodeDetector

# Initialize lists to store the detection results
qreader_correct = []
opencv_correct = []
pyzbar_correct = []

# Function to run a QR code detector with a timeout
def detect_with_timeout(detector_func, img, timeout=5):
    result_container = [None]  # Using a list to store the result as it's mutable
    
    def target():
        result_container[0] = detector_func(img)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print("  Detection timed out")
        return None  # Indicate timeout
    else:
        return result_container[0]

# Wrapper functions for each detector with timeout handling
def qreader_detect(img):
    return qreader_reader.detect_and_decode(image=img)

def opencv_detect(img):
    retval, decoded_info, _, _ = qcd.detectAndDecodeMulti(img)
    if retval and decoded_info:
        return decoded_info[0]
    return ""

def pyzbar_detect(img):
    results = decode(img)
    try:
        return [obj.data.decode('utf-8') for obj in results]
    except:
        return []

# Iterate through each row in the DataFrame
for row in df.itertuples():
    image_path = row.filename
    img = imread(image_path)

    # Check if the image was read successfully
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        qreader_correct.append(0)
        opencv_correct.append(0)
        pyzbar_correct.append(0)
        continue

    print(f"Processing {image_path}:")

    # Detect and decode using QReader with timeout
    qreader_result = detect_with_timeout(qreader_detect, img)
    qreader_match = correct_message in qreader_result if qreader_result is not None else False
    print(f"  QReader: {qreader_match}")

    # Detect and decode using OpenCV with timeout
    opencv_result = detect_with_timeout(opencv_detect, img)
    opencv_match = opencv_result == correct_message if opencv_result is not None else False
    print(f"  OpenCV: {opencv_match}")

    # Detect and decode using pyzbar with timeout
    pyzbar_results = detect_with_timeout(pyzbar_detect, img)
    pyzbar_match = correct_message in pyzbar_results if pyzbar_results is not None else False
    print(f"  pyzbar: {pyzbar_match}")

    # Append results
    qreader_correct.append(int(qreader_match))
    opencv_correct.append(int(opencv_match))
    pyzbar_correct.append(int(pyzbar_match))

# Add the results as new columns to the DataFrame
df['QReader Correct'] = qreader_correct
df['OpenCV Correct'] = opencv_correct
df['pyzbar Correct'] = pyzbar_correct

# Save the updated DataFrame to a new CSV file
output_csv = 'qr_code_reader_results.csv'
df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
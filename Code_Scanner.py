from qreader import QReader
from cv2 import imread
from pyzbar.pyzbar import decode
import pandas as pd
import cv2

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
    
    # Detect and decode using QReader
    qreader_result = qreader_reader.detect_and_decode(image=img)
    qreader_match = correct_message in qreader_result if qreader_result else False
    
    # Detect and decode using OpenCV
    opencv_result = ""
    retval, decoded_info, _, _ = qcd.detectAndDecodeMulti(img)
    if retval and decoded_info:
        opencv_result = decoded_info[0]  # Take the first detected QR code
    opencv_match = opencv_result == correct_message
    
    # Detect and decode using pyzbar
    pyzbar_results = decode(img)
    pyzbar_match = False
    try:
        decoded_messages = [obj.data.decode('utf-8') for obj in pyzbar_results]
        pyzbar_match = correct_message in decoded_messages
    except:
        pass  # Ignore decoding errors
    
    # Append results
    qreader_correct.append(int(qreader_match))
    opencv_correct.append(int(opencv_match))
    pyzbar_correct.append(int(pyzbar_match))
    
    # Print progress
    print(f"Processed {image_path}: QReader: {qreader_match}, OpenCV: {opencv_match}, pyzbar: {pyzbar_match}")

# Add the results as new columns to the DataFrame
df['QReader Correct'] = qreader_correct
df['OpenCV Correct'] = opencv_correct
df['pyzbar Correct'] = pyzbar_correct

# Save the updated DataFrame to a new CSV file
output_csv = 'qr_code_reader_results.csv'
df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
import os
import cv2
import csv
import re
import numpy as np
import logging
from paddleocr import PaddleOCR

# Initialize PaddleOCR with minimal parameters
# For colored text on black background in video frames
ocr = PaddleOCR(
    use_textline_orientation=True,  # Enable text angle classification (handles rotated text)
    lang='en'                        # English language
)

video_editing_folder_path = 'VIDEO_EDITING/'
csv_file_path = os.path.join(video_editing_folder_path, 'extracted_texts_english.csv')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

video_folder_path = 'fathmowillis'
video_editing_folder_path = 'VIDEO_EDITING'

def get_reel_number(filename):
    match = re.search(r"Video_(\d+)", filename)
    return int(match.group(1)) if match else None

def sort_csv():
    if not os.path.exists(csv_file_path):
        logging.warning(f"CSV file {csv_file_path} does not exist. Skipping sorting.")
        return

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        sorted_data = sorted(reader, key=lambda row: int(row['Reel Number']))

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Reel Number', 'Text'])
        writer.writeheader()
        writer.writerows(sorted_data)
        
def save_text_to_csv(reel_number, extracted_text):
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['Reel Number', 'Text'])

        writer.writerow([reel_number, extracted_text])
        logging.info(f"Saved text for reel {reel_number} to {csv_file_path}")
        
def extract_text_from_video(input_video_path, reel_number):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {input_video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error(f"Failed to read the first frame of video: {input_video_path}")
        cap.release()
        return

    white_area_text = extract_text_from_white_area(frame)
    print("white_area_text = ", white_area_text)

    unnecessary_words = ["W", "WA"]
    unnecessary_patterns = []
    remove_before = []
    remove_after = []
    custom_text = "If only there is a page dedicated"
    
    filtered_text = filter_text(white_area_text, unnecessary_words, unnecessary_patterns, remove_before, remove_after, custom_text)

    logging.info(f"Reel {reel_number}: Filtered Text: {filtered_text}")
    save_text_to_csv(reel_number, filtered_text)

    cap.release()


def extract_text_from_white_area(frame):
    frame_height, frame_width = frame.shape[:2]
    print(f"Complete frame size: Width = {frame_width}, Height = {frame_height}")
    
    min_x = 0
    max_x = frame_width
    min_y = 0
    max_y = frame_height
    print("min_x = {}, min_y = {}, max_x = {}, max_y = {}".format(min_x,min_y,max_x,max_y))
    # Optionally save debug frame (commented out for headless environments)
    # output_dir = "VIDEO_EDITING" 
    # os.makedirs(output_dir, exist_ok=True)
    # if min_x < max_x and min_y < max_y: 
    #     text_region = frame[min_y:max_y, min_x:max_x] 
    #     output_filename = os.path.join(output_dir, f"text_frame.png") 
    #     cv2.imwrite(output_filename, text_region) 
    #     print(f"Saved: {output_filename}") 

    # Note: cv2.imshow() and cv2.waitKey() removed - not compatible with headless environments (Colab)

    # Crop the region of interest
    white_area = frame[min_y:max_y, min_x:max_x]
    
    # PaddleOCR works with BGR images (OpenCV default), no need to convert
    # For colored text on black background, the contrast is already good
    
    # Run OCR inference using predict() method (PaddleOCR 3.x API)
    result = ocr.predict(white_area)

    # Extract text from results
    white_area_text = ""
    if result and len(result) > 0:
        for res in result:
            # The result is an OCRResult object that acts like a dict
            # It has 'rec_texts' (list of detected texts) and 'rec_scores' (list of confidence scores)
            if isinstance(res, dict) or hasattr(res, '__getitem__'):
                rec_texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
                rec_scores = res.get('rec_scores', []) if isinstance(res, dict) else getattr(res, 'rec_scores', [])
                
                # Iterate through detected texts and their confidence scores
                for text, score in zip(rec_texts, rec_scores):
                    # Only include text with confidence > 0.5 to filter out noise
                    if score > 0.5:
                        white_area_text += text + " "
                        print(f"Detected: '{text}' (confidence: {score:.2f})")

    print(f"Final extracted text: '{white_area_text.strip()}'")
    return white_area_text.strip()


def filter_text(complete_text, unnecessary_words=None, unnecessary_patterns=None, remove_before=None, remove_after=None, fallback_text=""):
    filtered_text = complete_text    
    filtered_text = re.sub(r'[^\w\s]', '', filtered_text)

    if unnecessary_words:
        words = filtered_text.split()
        filtered_text = " ".join(word for word in words if word not in unnecessary_words)

    if unnecessary_patterns:
        lines = filtered_text.splitlines()
        filtered_text = "\n".join(line for line in lines if not any(re.search(pattern, line) for pattern in unnecessary_patterns))

    if remove_before:
        for word in remove_before:
            match = re.search(rf'\b{word}\b', filtered_text, re.IGNORECASE)
            if match:
                filtered_text = filtered_text[match.start():]
                break

    if remove_after:
        for word in remove_after:
            match = re.search(rf'\b{word}\b', filtered_text, re.IGNORECASE)
            if match:
                filtered_text = filtered_text[:match.end()]
                break

    if not filtered_text.strip():
        logging.info("Filtered text is empty, replacing with fallback text.")
        filtered_text = fallback_text
            
    return filtered_text.strip()
    
def process_video(input_video_path, reel_number):
    extract_text_from_video(input_video_path, reel_number)

def get_input_video(reel_number):
    for filename in os.listdir(video_folder_path):
        if filename.startswith("Video") and filename.endswith(f"_{reel_number}.mp4"):
            input_video_path = os.path.join(video_folder_path, filename) 
            logging.info(f'Processing {filename} as reel_{reel_number}')
            return input_video_path

def process_all_videos():
    for reel_number in range(520, 1063):
        print(f"Processing reel number: {reel_number}")
        input_video_path = get_input_video(reel_number)
        process_video(input_video_path, reel_number)

process_all_videos()
sort_csv()

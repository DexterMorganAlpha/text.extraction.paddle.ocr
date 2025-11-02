import os
import cv2
import csv
import re
import numpy as np
import logging
import pytesseract

# Path to Tesseract OCR executable (update this to your system's path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
video_editing_folder_path = 'TEXT/'
csv_file_path = os.path.join(video_editing_folder_path, 'extracted_texts.csv')
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Folder paths and common parameters
video_folder_path = 'TEXT/VIDEOS/'

def get_reel_number(filename):
    match = re.search(r"Video_(\d+)", filename)
    return int(match.group(1)) if match else None

def sort_csv():
    """
    Sort the CSV by reel_number and overwrite the file with the sorted data.
    """
    if not os.path.exists(csv_file_path):
        logging.warning(f"CSV file {csv_file_path} does not exist. Skipping sorting.")
        return

    # Read and sort the data
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        sorted_data = sorted(reader, key=lambda row: int(row['Reel Number']))

    # Write the sorted data back to the file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Reel Number', 'Text'])
        writer.writeheader()  # Write header
        writer.writerows(sorted_data)
        
def save_text_to_csv(reel_number, extracted_text):
    """
    Save the reel_number and extracted text to a CSV file.
    If the file does not exist, it will create a new one with headers.
    """
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers if the file is new
        if not file_exists:
            writer.writerow(['Reel Number', 'Text'])

        # Write the data
        writer.writerow([reel_number, extracted_text])
        logging.info(f"Saved text for reel {reel_number} to {csv_file_path}")
        
def extract_text_from_video(input_video_path, reel_number):
    """
    Extract text from the first frame and filter out text present in the cropped area.
    Save the filtered text to a CSV file.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {input_video_path}")
        return

    ret, frame = cap.read()  # Read only the first frame
    if not ret:
        logging.error(f"Failed to read the first frame of video: {input_video_path}")
        cap.release()
        return

    # Extract text from the white area
    white_area_text = extract_text_from_white_area(frame)
    print("white_area_text = ", white_area_text)
    
    # Define unnecessary words and patterns
    # unnecessary_words = ["Follow", "howthingsformed"]  # Example words to filter out
    # # unnecessary_patterns = [r"http\S+", r"example"]  # Example patterns to exclude URLs or specific keywords
    # unnecessary_patterns = ["If only there was a page that posts how things are made in the world",
    #                         "If there is a page dedicated to how things are formed",
    #                         "If only there was a page dedicated to how things are formed",
    #                         "If only there is a page dedicated to how things are formed"
    #                         ]  # Example patterns to exclude URLs or specific keywords
    # remove_before = ["If", "HOW", "How"]
    # remove_after = ["formed", "world", "made"]
    # custom_text = "If only there is a page dedicated to most beautiful girls."
    
    # Define unnecessary words and patterns
    unnecessary_words = ["W", "WA"]  # Example words to filter out
    # unnecessary_patterns = [r"http\S+", r"example"]  # Example patterns to exclude URLs or specific keywords
    unnecessary_patterns = []  # Example patterns to exclude URLs or specific keywords
    remove_before = []
    remove_after = []
    custom_text = "If only there is a page dedicated to most beautiful girls."
    
    # Filter cropped text and unnecessary content
    filtered_text = filter_text(white_area_text, unnecessary_words, unnecessary_patterns, remove_before, remove_after,custom_text)

    logging.info(f"Reel {reel_number}: Filtered Text: {filtered_text}")
    
    # Save the white area text to the CSV
    save_text_to_csv(reel_number, filtered_text)

    cap.release()


def extract_text_from_white_area(frame):
    """
    Detect the white area in the frame and extract text using OCR.
    Args:
        frame (np.array): The input frame from which the text needs to be extracted.
    Returns:
        str: Extracted text from the white area.
    """
    logging.debug("Extracting text from white area.")
    # Convert frame to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # Define thresholds for white color
    # lower_white = np.array([0, 0, 200], dtype=np.uint8)
    # upper_white = np.array([255, 55, 255], dtype=np.uint8)

    # # Create a mask for white areas
    # white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # Convert frame to grayscale 
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Apply adaptive thresholding to enhance text detection 
    # blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    # blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    # blurred = cv2.medianBlur(blurred, 5)
    # edges = cv2.Canny(blurred, 50, 250)
    # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    # cv2.imshow("Thresholded", thresh)
    # cv2.waitKey(3000)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the white areas
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     logging.info("No white area detected in the frame.")
    #     return ""

    # # Find the largest white contour (assuming it corresponds to the white area of interest)
    # largest_contour = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(largest_contour)
    # Use Pytesseract to detect text bounding boxes 
    # data = pytesseract.image_to_data(thresh, config="--psm 6", output_type=pytesseract.Output.DICT) 

    # Convert confidence values safely, ignoring errors 
    # conf_values = []
    # for conf in data["conf"]:
    #     try:
    #         conf_values.append(int(conf))
    #     except ValueError:
    #         conf_values.append(0)  # Default to 0 if conversion fails
            
    # # Initialize refined bounding box variables for text 
    # min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0
    # # Iterate through detected text regions 
    # for i in range(len(data["text"])):
    #     if conf_values[i] > 40 and data["text"][i].strip(): # Confidence threshold and non-empty text
    #         x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i] 
		
    #         # Ensure detection is within the upper white area (above the video) 
        
    #         if y < frame.shape[0] // 4: 
    #             min_x = min(min_x, x) 
    #             min_y = min(min_y, y) 
    #             max_x = max(max_x, x + w) 
    #             max_y = max(max_y, y + h) # Extract and save the text region if detected
    
    # Manual Modifications - Get complete frame size
    frame_height, frame_width = frame.shape[:2]
    print(f"Complete frame size: Width = {frame_width}, Height = {frame_height}")
    
    min_x = 0
    max_x = frame_width
    min_y = 0
    max_y = frame_height
    print("min_x = {}, min_y = {}, max_x = {}, max_y = {}".format(min_x,min_y,max_x,max_y))
    output_dir = "VIDEO_EDITING" 
    os.makedirs(output_dir, exist_ok=True)
    if min_x < max_x and min_y < max_y: 
        text_region = frame[min_y:max_y, min_x:max_x] 
        output_filename = os.path.join(output_dir, f"text_frame.png") 
        cv2.imwrite(output_filename, text_region) 
        print(f"Saved: {output_filename}") 

    # Draw the detected bounding box for debugging
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    cv2.imshow("Final Detected Video Area", debug_frame)
    cv2.waitKey(1000)
    
    # Crop the white area from the frame
    white_area = frame[min_y:max_y, min_x:max_x]

    # Convert the white area to grayscale
    # gray_white_area = cv2.cvtColor(white_area, cv2.COLOR_BGR2GRAY)

    # Apply OCR to extract text
    white_area_text = pytesseract.image_to_string(white_area, lang='eng').strip()
    # logging.info(f"Extracted text from white area: {white_area_text}")
    
    return white_area_text

def filter_text(complete_text, unnecessary_words=None, unnecessary_patterns=None, remove_before=None, remove_after=None, fallback_text=""):
    """
    Filter out cropped area text, unnecessary characters, words, and sentences from the complete text.
    Args:
        complete_text (str): Full text extracted from the first frame.
        cropped_text (str): Text extracted from the cropped area.
        unnecessary_words (list): List of words to remove from the final text.
        unnecessary_patterns (list): List of regex patterns to filter out sentences.
        remove_before (list): List of words; remove everything before any of these words.
        remove_after (list): List of words; remove everything after any of these words.
        fallback_text (str): Custom text to replace if the filtered text becomes empty.

    Returns:
        str: Cleaned and filtered text.
    """
    # if not cropped_text:
        # filtered_text = complete_text
    # else:
    #     # Break texts into sets of lines for comparison
    #     complete_lines = set(complete_text.splitlines())
    #     cropped_lines = set(cropped_text.splitlines())

    #     # Subtract cropped lines from complete lines
    #     filtered_lines = complete_lines - cropped_lines

    #     # Reconstruct the text
    #     filtered_text = "\n".join(filtered_lines)

    filtered_text = complete_text
    
    # Step 1: Remove unnecessary characters (e.g., special characters)
    filtered_text = re.sub(r'[^\w\s]', '', filtered_text)  # Keep only words and spaces

    # Step 2: Remove unnecessary words
    if unnecessary_words:
        # print("unnecessary words = ",unnecessary_words)
        words = filtered_text.split()
        # for word in words:
        #     print("word = ",word)
        filtered_text = " ".join(word for word in words if word not in unnecessary_words)

    # Step 3: Remove unnecessary sentences based on patterns
    if unnecessary_patterns:
        lines = filtered_text.splitlines()
        filtered_text = "\n".join(line for line in lines if not any(re.search(pattern, line) for pattern in unnecessary_patterns))
    
    # Step 4: Remove everything before specified words
    if remove_before:
        for word in remove_before:
            match = re.search(rf'\b{word}\b', filtered_text, re.IGNORECASE)
            if match:
                filtered_text = filtered_text[match.start():]
                break  # Stop after the first match

    # Step 5: Remove everything after specified words
    if remove_after:
        for word in remove_after:
            match = re.search(rf'\b{word}\b', filtered_text, re.IGNORECASE)
            if match:
                filtered_text = filtered_text[:match.end()]
                break  # Stop after the first match

    # Step 6: Check if text is empty and replace with fallback text
    if not filtered_text.strip():
        logging.info("Filtered text is empty, replacing with fallback text.")
        filtered_text = fallback_text
            
    return filtered_text.strip()
    
# Update the process_video function to include text extraction
def process_video(input_video_path, reel_number):

    # Extract text and save to CSV
    extract_text_from_video(input_video_path, reel_number)

def get_input_video(reel_number):
    ## Loop over the video folder and get the input video path
    for filename in os.listdir(video_folder_path):
        if filename.startswith("Video") and filename.endswith(f"_{reel_number}.mp4"):
            input_video_path = os.path.join(video_folder_path, filename) 
            logging.info(f'Processing {filename} as reel_{reel_number}')

    return input_video_path
# counter_file = 'VIDEO_EDITING/../counter.txt'
# def get_reel_number():
#     """Read the current counter value from the file, or initialize it."""
#     if os.path.exists(counter_file):
#         with open(counter_file, 'r') as file:
#             return int(file.read())
#     return 0

def process_all_videos():
    # for filename in os.listdir(video_folder_path):
    #     if filename.startswith("Video_") and filename.endswith(".mp4"):
    #         reel_number = get_reel_number(filename)
    #         if reel_number is None:
    #             continue

    #         input_video_path = os.path.join(video_folder_path, filename)

    #         logging.info(f'Processing {filename} as reel_{reel_number}')
    #         process_video(input_video_path, reel_number)
            
    # # Getting the reel number from counter file.
    # reel_number = get_reel_number()
    # print(f"Reel Number : {reel_number}")
    
    # # Remove the previous day's reel before starting today's processing
    # # remove_previous_reel(reel_number)
    for reel_number in range(1, 894):
        # Get the input video from video folder.
        print(f"Processing reel number: {reel_number}")
        input_video_path = get_input_video(reel_number)
        
        # Process the input video
        process_video(input_video_path, reel_number)

# Run the batch processing
process_all_videos()
sort_csv()
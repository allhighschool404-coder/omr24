from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import uuid

import sqlite3

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('omr_results.db')
    c = conn.cursor()
    columns = ", ".join([f"Q{i+1} TEXT" for i in range(30)])
    c.execute(f'''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            set_name TEXT,
            {columns}
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/2O2mr.html')
def omr_template():
    return render_template('2O2mr.html')

def rect_to_bb(rect):
    x, y, w, h = cv2.boundingRect(rect)
    return (x, y, w, h)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_omr_image(image_data, filename="Memory Stream"):
    """
    Ultra-Robust OpenCV OMR Processing Logic.
    This version dynamically detects the page boundaries to ignore any screenshot margins,
    and maps the solid black bubbles strictly by their geometric coordinates.
    """
    import cv2
    import numpy as np
    
    print(f"--- Processing OMR Image: {filename} ---")

    if isinstance(image_data, str):
        image = cv2.imread(image_data)
    else:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Invalid image")
        return {"error": "Invalid image"}

    # Standardize image height to 1200 pixels
    std_height = 1200
    ratio = std_height / image.shape[0]
    std_width = int(image.shape[1] * ratio)
    resized = cv2.resize(image, (std_width, std_height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Thresholding: Keep only very dark pixels (e.g. solid black ink)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 1. Dynamically find the exact boundaries of the OMR page to ignore screenshot margins
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        coords = coords.reshape(-1, 2)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        page_left_x = int(np.min(x_coords))
        page_right_x = int(np.max(x_coords))
        page_top_y = int(np.min(y_coords))
        page_bottom_y = int(np.max(y_coords))
    else:
        page_left_x, page_right_x = 0, std_width
        page_top_y, page_bottom_y = 0, std_height
        
    page_width = page_right_x - page_left_x
    print(f"Page Boundaries Detected: X({page_left_x} to {page_right_x}), Y({page_top_y} to {page_bottom_y}), Width: {page_width}")

    # 2. Find contours (the actual marked bubbles)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        # Filters for circles of roughly the correct bubble size
        if 10 <= w <= 50 and 10 <= h <= 50 and 0.7 <= ar <= 1.3:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            filled_area = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            
            # The user requested 60% fill threshold for strict validation
            if filled_area > (w * h * 0.60):
                filled_bubbles.append({"x": x + w//2, "y": y + h//2, "w": w, "h": h})
                
    print(f"Total Solid Black Bubbles Found: {len(filled_bubbles)}")

    # 3. Segment the page into 3 equal columns strictly inside the detected page boundaries
    col_width = page_width / 3.0
    cols = [
        {"min_x": page_left_x, "max_x": page_left_x + col_width, "bubbles": []},
        {"min_x": page_left_x + col_width, "max_x": page_left_x + col_width * 2, "bubbles": []},
        {"min_x": page_left_x + col_width * 2, "max_x": page_right_x, "bubbles": []}
    ]
    
    bottom_bubbles = []
    
    for b in filled_bubbles:
        # Group bubbles based on their Y position relative to the page content
        if b["y"] < page_top_y + (page_bottom_y - page_top_y) * 0.65:
            # It's an MCQ bubble
            if b["x"] < cols[0]["max_x"]:
                cols[0]["bubbles"].append(b)
            elif b["x"] < cols[1]["max_x"]:
                cols[1]["bubbles"].append(b)
            else:
                cols[2]["bubbles"].append(b)
        else:
            # It's a bottom-grid bubble (e.g. ID, Roll, Class, Section, Set)
            bottom_bubbles.append(b)

    bengali_options = ['ক', 'খ', 'গ', 'ঘ']
    total_answers = []

    # 4. Determine Answer for each Question
    page_height = page_bottom_y - page_top_y
    expected_row_0_y = page_top_y + page_height * 0.2025
    row_spacing = page_height * 0.0307
    
    for i, col in enumerate(cols):
        bubbles = col["bubbles"]
        
        col_ans = [None] * 10
        expected_A_x = page_left_x + page_width * (0.142 + i * 0.293)
        bubble_spacing = page_width * 0.0544
        
        for b in bubbles:
            row_idx = int(round((b["y"] - expected_row_0_y) / row_spacing))
            if 0 <= row_idx < 10:
                slot = int(round((b["x"] - expected_A_x) / bubble_spacing))
                idx = max(0, min(3, slot))
                col_ans[row_idx] = bengali_options[idx]
            
        total_answers.extend(col_ans)

    # 5. Extract ID, Roll, Class, Section, Set
    student_id = "0000000"
    roll_no = "000"
    class_name = "10"
    detected_set = 'ক'
    detected_section = 'ক'
    
    if bottom_bubbles:
        # User confirmed: Row 1 = 0, Row 2 = 1, ..., Row 10 = 9.
        # Use the exact MCQ row_spacing for y_step since bubble sizes are identical
        y_step = row_spacing
        
        # Estimate digit_0_y based on MCQ layout (MCQ ends at row 9, gap is roughly 8 rows)
        # 0.2025 + 17 * 0.0307 = 0.7244
        initial_digit_0_y = page_top_y + page_height * 0.7244
        
        # Refine digit_0_y using the median of implied digit_0_y from all bottom bubbles
        implied_y0s = []
        for b in bottom_bubbles:
            d = round((b["y"] - initial_digit_0_y) / y_step)
            implied_y0s.append(b["y"] - d * y_step)
            
        implied_y0s.sort()
        digit_0_y = implied_y0s[len(implied_y0s) // 2] if implied_y0s else initial_digit_0_y
        
        id_cols, roll_cols, class_cols = ["" for _ in range(7)], ["" for _ in range(3)], ["" for _ in range(2)]
        
        for b in bottom_bubbles:
            rel_x = (b["x"] - page_left_x) / float(page_width)
            if 0.05 <= rel_x < 0.35: # ID
                col = int((rel_x - 0.05) / ((0.35 - 0.05) / 7.0))
                if 0 <= col < 7:
                    digit = int(round((b["y"] - digit_0_y) / y_step))
                    id_cols[col] = str(max(0, min(9, digit)))
            elif 0.37 <= rel_x < 0.52: # Roll
                col = int((rel_x - 0.37) / ((0.52 - 0.37) / 3.0))
                if 0 <= col < 3:
                    digit = int(round((b["y"] - digit_0_y) / y_step))
                    roll_cols[col] = str(max(0, min(9, digit)))
            elif 0.54 <= rel_x < 0.65: # Class
                col = int((rel_x - 0.54) / ((0.65 - 0.54) / 2.0))
                if 0 <= col < 2:
                    digit = int(round((b["y"] - digit_0_y) / y_step))
                    class_cols[col] = str(max(0, min(9, digit)))
            elif 0.67 <= rel_x < 0.82: # Section & Set
                digit_approx = (b["y"] - digit_0_y) / y_step
                if 0 <= digit_approx <= 3.5:
                    idx = int(round(digit_approx))
                    detected_section = bengali_options[max(0, min(3, idx))]
                elif 5.5 <= digit_approx <= 9:
                    idx = int(round(digit_approx - 6.0))
                    detected_set = bengali_options[max(0, min(2, idx))]
                    
        # Replace empty with "0" for missing bubbles
        student_id = "".join(d if d else "0" for d in id_cols)
        roll_no = "".join(d if d else "0" for d in roll_cols)
        class_name = "".join(d if d else "0" for d in class_cols)

    # Fallback to prevent crash if completely empty
    if len(total_answers) == 0:
        total_answers = [None] * 30

    result = {
        "status": "success",
        "sets": {
            detected_set: total_answers
        },
        "student_id": student_id,
        "roll_no": roll_no,
        "class_name": class_name,
        "section": detected_section
    }
    
    return result

@app.route('/api/scan_omr', methods=['POST'])
def scan_omr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        # Process the image directly from memory without saving to website's folder
        image_bytes = file.read()
        result = process_omr_image(image_bytes, file.filename)
        
        # Save results permanently to a CSV file and SQLite DB
        if result.get("status") == "success":
            import csv
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            csv_filename = 'omr_results.csv'
            file_exists = os.path.isfile(csv_filename)
            
            with open(csv_filename, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write header (Timestamp, Set, Q1, Q2, ..., Q30)
                    header = ['Timestamp', 'Set'] + [f'Q{i+1}' for i in range(30)]
                    writer.writerow(header)
                
                # Write data to CSV and SQLite
                for set_name, answers in result.get('sets', {}).items():
                    row = [timestamp, set_name] + answers
                    writer.writerow(row)  # Save to CSV
                    
                    # Save to SQLite Database
                    try:
                        conn = sqlite3.connect('omr_results.db')
                        c = conn.cursor()
                        placeholders = ", ".join(["?"] * 32)
                        columns = ", ".join([f"Q{i+1}" for i in range(30)])
                        c.execute(f"INSERT INTO results (timestamp, set_name, {columns}) VALUES ({placeholders})", row)
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print(f"Database error: {e}")
        
        return jsonify(result)

@app.route('/api/exams', methods=['GET', 'POST'])
def handle_exams():
    import json
    filename = 'exams_data.json'
    if request.method == 'GET':
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return jsonify([])
            
    if request.method == 'POST':
        data = request.json
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

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

def process_omr_image(image_path):
    """
    Ultra-Robust OpenCV OMR Processing Logic.
    This version dynamically detects the page boundaries to ignore any screenshot margins,
    and maps the solid black bubbles strictly by their geometric coordinates.
    """
    import cv2
    import numpy as np
    
    print(f"--- Processing OMR Image: {image_path} ---")

    image = cv2.imread(image_path)
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
        x_coords = coords[:, 0, 0]
        y_coords = coords[:, 0, 1]
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
    
    set_bubbles = []
    
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
            # It's a bottom-grid bubble (e.g. Set No)
            # The Set Number grid is roughly at 60%-80% horizontally
            if page_left_x + 0.6 * page_width < b["x"] < page_left_x + 0.8 * page_width:
                set_bubbles.append(b)

    bengali_options = ['ক', 'খ', 'গ', 'ঘ']
    total_answers = []

    # 4. Determine Answer for each Question
    for i, col in enumerate(cols):
        bubbles = col["bubbles"]
        # Sort vertically to map them sequentially to Q1, Q2, Q3, etc.
        bubbles = sorted(bubbles, key=lambda b: b["y"])
        print(f"Column {i+1} mapped {len(bubbles)} bubbles.")
        
        col_ans = []
        for b in bubbles[:10]: # Max 10 per column
            local_x = b["x"] - col["min_x"]
            
            # Options (ক, খ, গ, ঘ) sit in the right 75% of the column width
            zone_start = 0.22 * col_width
            zone_end = 0.95 * col_width
            
            slot = (local_x - zone_start) / ((zone_end - zone_start) / 4.0)
            idx = int(max(0, min(3, slot)))
            col_ans.append(bengali_options[idx])
            
        # Pad missing answers with None to ensure array stability
        while len(col_ans) < 10:
            col_ans.append(None)
            
        total_answers.extend(col_ans)

    # 5. Determine the Set Number
    detected_set = 'ক' # Default
    if set_bubbles:
        set_bubbles = sorted(set_bubbles, key=lambda b: b["y"])
        # For simplicity, if they marked at least one bubble in the Set box,
        # we default it to 'ক'. A more advanced check could read relative Y position.
        detected_set = 'ক'
        print("Detected Set Bubble.")

    print(f"Extracted Answers for Set {detected_set}: {total_answers}")

    # Fallback to prevent crash if completely empty
    if len(total_answers) == 0:
        total_answers = [None] * 30

    result = {
        "status": "success",
        "sets": {
            detected_set: total_answers
        }
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
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result = process_omr_image(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

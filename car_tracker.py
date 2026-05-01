import cv2
import csv
import sys
import concurrent.futures

# Configuration
VIDEO_INPUT_PATH = '1900-151662242.mp4'
VIDEO_OUTPUT_PATH = 'tracked_output.mp4'
CSV_OUTPUT_PATH = 'tracking_data.csv'

CAR_1_TEMPLATE = 'image1.png'
CAR_2_TEMPLATE = 'image2.png'

TRIM_PERCENT = 0.0 

def check_single_scale(scale, template, gray_frame):
    """Worker function to check a single template scale."""
    # Calculate templates dimensions based on the provided scale
    width = int(template.shape[1] * scale)
    height = int(template.shape[0] * scale)
    
    # Check if dimensions are valid for the frame
    if width <= 0 or height <= 0 or width > gray_frame.shape[1] or height > gray_frame.shape[0]:
        return None, -1
        
    # Resize and match the template
    resized_template = cv2.resize(template, (width, height))
    res = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    # Apply the configurable trim
    trim_x = int(width * TRIM_PERCENT) 
    trim_y = int(height * TRIM_PERCENT)
    
    final_x = max_loc[0] + trim_x
    final_y = max_loc[1] + trim_y
    final_w = max(1, width - (trim_x * 2))
    final_h = max(1, height - (trim_y * 2))
    
    match_bbox = (final_x, final_y, final_w, final_h)
    return match_bbox, max_val

def get_initial_bbox(frame, template_path):
    """Finds the template using Multi-Scale matching across multiple CPU cores."""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not read template image {template_path}")
        sys.exit()
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    best_match = None
    best_val = -1
    
    # Checking sizes from 60% to 140% to prevent it from matching tiny sub-features
    scales = [x / 10.0 for x in range(6, 15)] 
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_single_scale, scale, template, gray_frame) for scale in scales]
        
        for future in concurrent.futures.as_completed(futures):
            match_bbox, conf_val = future.result()
            if conf_val > best_val:
                best_val = conf_val
                best_match = match_bbox
            
    return best_match, best_val

def create_tracker():
    """Safely initializes the KCF tracker."""
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    else:
        print("Error: KCF Tracker not found. Please run: pip install opencv-contrib-python")
        sys.exit()

def track_multiple_cars():
    video = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps: fps = 30.0 
    
    delay = int(1000 / fps) 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

    print("Scanning video to find initial high-confidence matches...")
    MATCH_THRESHOLD = 0.70 
    
    bbox1, bbox2 = None, None
    frame_num = 0
    starting_frame = None
    
    while True:
        ok, frame = video.read()
        if not ok:
            print("Error: Reached end of video without finding the templates.")
            return
            
        frame_num += 1
        
        if frame_num % 3 != 0:
            continue
            
        b1, conf1 = get_initial_bbox(frame, CAR_1_TEMPLATE)
        b2, conf2 = get_initial_bbox(frame, CAR_2_TEMPLATE)
        
        if conf1 >= MATCH_THRESHOLD and conf2 >= MATCH_THRESHOLD:
            print(f"Match found at Frame {frame_num}!")
            bbox1 = b1
            bbox2 = b2
            starting_frame = frame
            break

    tracker1 = create_tracker()
    tracker2 = create_tracker()
    
    tracker1.init(starting_frame, bbox1)
    tracker2.init(starting_frame, bbox2)

    with open(CSV_OUTPUT_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV header for tracking data
        csv_writer.writerow(['Frame Number', 'Car1_X', 'Car1_Y', 'Car1_W', 'Car1_H', 'Car2_X', 'Car2_Y', 'Car2_W', 'Car2_H'])

        # Write initial starting positions to CSV
        csv_writer.writerow([frame_num] + list(bbox1) + list(bbox2))
        out_video.write(starting_frame)
        
        print("Displaying real-time tracking... (Press 'q' to stop)")
        
        while True:
            # Read subsequent frames
            ok, frame = video.read()
            if not ok:
                break 
            
            frame_num += 1

            # Update bounding boxes with tracker
            success1, bbox1 = tracker1.update(frame)
            success2, bbox2 = tracker2.update(frame)

            if success1:
                # Car 1 tracked successfully - parse coords and draw box
                x1, y1, w1, h1 = [int(v) for v in bbox1]
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2) 
                cv2.putText(frame, "Car 1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                car1_data = [x1, y1, w1, h1]
            else:
                # Car 1 lost - record dummy data
                car1_data = ['Lost', 'Lost', 'Lost', 'Lost']

            if success2:
                # Car 2 tracked successfully - parse coords and draw box
                x2, y2, w2, h2 = [int(v) for v in bbox2]
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2) 
                cv2.putText(frame, "Car 2", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                car2_data = [x2, y2, w2, h2]
            else:
                # Car 2 lost - record dummy data
                car2_data = ['Lost', 'Lost', 'Lost', 'Lost']

            # Log frame bounds to CSV and write video frame
            csv_writer.writerow([frame_num] + car1_data + car2_data)
            out_video.write(frame)
            
            # Show live output window
            cv2.imshow("Multi-Car Tracking (Real-Time)", frame)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    video.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved video to {VIDEO_OUTPUT_PATH} and data to {CSV_OUTPUT_PATH}")

if __name__ == '__main__':
    track_multiple_cars()
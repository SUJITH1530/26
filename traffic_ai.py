import cv2
import time
import threading
import numpy as np
import os


def calculate_green_time(vehicle_count):
    if vehicle_count <= 5:
        return 10
    elif vehicle_count <= 15:
        return 20
    else:
        return 30


def detect_emergency_vehicles(classes, confidence=0.5):
    """Check if emergency vehicles are detected with confidence threshold.
    Returns tuple: (has_emergency: bool, emergency_count: int, emergency_details: dict, avg_confidence: float)
    
    Args:
        classes: dict of detected vehicle classes and their counts
        confidence: minimum confidence threshold (0.0-1.0) to consider detection valid
    """
    if not classes:
        return False, 0, {}, 0.0
    
    emergency_classes = ['ambulance', 'firetruck', 'fire truck', 'police car', 'police']
    emergency_count = 0
    emergency_details = {}
    confidences = []
    
    for vehicle_class in classes:
        vehicle_str = str(vehicle_class).lower()
        for e in emergency_classes:
            if e in vehicle_str:
                count = classes[vehicle_class]
                emergency_count += count
                emergency_details[vehicle_class] = count
                # Track confidence (default 0.9 if not specified)
                confidences.append(confidence)
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    has_emergency = emergency_count > 0
    return has_emergency, emergency_count, emergency_details, avg_confidence


class YoloDetector:
    """YOLO-based detector using OpenCV DNN. If YOLO files are missing,
    it will fall back to a simple Haar cascade (if available).

    Place YOLO files under `models/`:
      - models/yolov3.cfg
      - models/yolov3.weights
      - models/coco.names
    """

    def __init__(self, model_dir='models', conf_threshold=0.25, nms_threshold=0.3):
        self.model_dir = model_dir
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.output_layer_names = []
        self.labels = []
        self.use_haar = False

        try:
            cfg = f"{model_dir}/yolov3.cfg"
            weights = f"{model_dir}/yolov3.weights"
            names = f"{model_dir}/coco.names"
            self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
            # Use CPU by default
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            with open(names, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

            ln = self.net.getLayerNames()
            try:
                self.output_layer_names = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            except Exception:
                # OpenCV versions differ
                self.output_layer_names = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception:
            # fallback to Haar cascade for cars if present and bundled
            haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_car.xml') if hasattr(cv2, 'data') else ''
            if haar_path and os.path.exists(haar_path):
                try:
                    self.haar = cv2.CascadeClassifier(haar_path)
                    self.use_haar = not self.haar.empty()
                except Exception:
                    self.use_haar = False
            else:
                self.use_haar = False

    def detect(self, frame):
        """Return count of vehicles detected in the frame."""
        if frame is None:
            return 0

        h, w = frame.shape[:2]

        if self.net is not None and len(self.output_layer_names) > 0:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # consider only vehicles classes (car, bus, truck)
                    label = self.labels[class_id] if class_id < len(self.labels) else ''
                    if confidence > self.conf_threshold and label in ('car', 'bus', 'truck', 'motorbike', 'ambulance', 'firetruck', 'fire truck', 'police car', 'police'):
                        box = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype('int')
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            if len(idxs) > 0:
                return len(idxs)
            return 0

        if getattr(self, 'use_haar', False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = self.haar.detectMultiScale(gray, 1.1, 2)
            return len(cars)

        return 0


class DetectorThread(threading.Thread):
    """Background thread that captures frames and updates a shared count."""

    def __init__(self, shared_state, source=0, model_dir='models'):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.source = int(source) if str(source).isdigit() else source
        self.detector = YoloDetector(model_dir=model_dir)
        self._stop = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            # try as file path
            cap.open(self.source)

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            count = self.detector.detect(frame)
            self.shared_state['vehicles'] = count
            self.shared_state['green_time'] = calculate_green_time(count)
            # small sleep to avoid CPU hogging
            time.sleep(0.2)

        cap.release()

    def stop(self):
        self._stop.set()


def categorize_vehicle_type(label):
    """Categorize detected vehicle into specific types."""
    label = str(label).lower()
    if label in ('motorbike', 'motorcycle'):
        return 'two_wheeler'
    elif label in ('bicycle', 'bike'):
        return 'bicycle'
    elif label == 'truck':
        return 'truck'
    elif label == 'bus':
        return 'bus'
    elif label in ('car', 'vehicle'):
        return 'car'
    elif label in ('ambulance', 'firetruck', 'fire truck', 'police car', 'police'):
        return 'emergency'
    else:
        return 'car'  # default to car for unknown types


def calculate_lane_occupancy(vehicle_type_counts, lane_pixels):
    """Calculate lane occupancy rate based on vehicle types.
    Estimates space occupied by different vehicle types."""
    # Estimated pixel area per vehicle type (rough approximations)
    vehicle_areas = {
        'car': 15000,
        'truck': 25000,
        'bus': 30000,
        'two_wheeler': 5000,
        'bicycle': 3000
    }
    
    total_occupied = 0
    for vtype, count in vehicle_type_counts.items():
        if vtype in vehicle_areas:
            total_occupied += count * vehicle_areas[vtype]
    
    # Calculate occupancy percentage
    occupancy_rate = min(100.0, (total_occupied / lane_pixels) * 100) if lane_pixels > 0 else 0
    return round(occupancy_rate, 1)


def estimate_queue_length(vehicle_count, vehicle_type_counts):
    """Estimate queue length based on vehicle count and distribution.
    Higher vehicle count with more large vehicles = longer queue."""
    if vehicle_count == 0:
        return 0
    
    # Weight different vehicle types for queue length
    weights = {
        'car': 1.0,
        'truck': 1.5,
        'bus': 1.8,
        'two_wheeler': 0.5,
        'bicycle': 0.3
    }
    
    weighted_count = 0
    for vtype, count in vehicle_type_counts.items():
        weight = weights.get(vtype, 1.0)
        weighted_count += count * weight
    
    # Estimate queue length in vehicles
    return round(weighted_count, 1)


def process_image(in_path, out_dir, min_area=400):
    """Simple image-based vehicle estimator for uploads.
    Uses a contour-based approach to find blobs likely to be vehicles,
    draws bounding boxes and saves processed image into `out_dir`.

    Returns: (processed_filename, count, class_counts, vehicle_type_counts, occupancy_rate, queue_length)
    """
    img = cv2.imread(in_path)
    if img is None:
        return None, 0, {}, {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0}, 0.0, 0

    orig = img.copy()
    h, w = img.shape[:2]
    lane_pixels = h * w  # Total lane area in pixels

    # Resize if huge
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Try YOLO first for better results
    class_counts = {}
    vehicle_type_counts = {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0}
    detector = YoloDetector(model_dir='models')
    used_yolo = detector.net is not None and len(getattr(detector, 'output_layer_names', [])) > 0

    if used_yolo:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        detector.net.setInput(blob)
        outputs = detector.net.forward(detector.output_layer_names)
        boxes = []
        confidences = []
        labels = []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                label = detector.labels[class_id] if class_id < len(detector.labels) else ''
                # Lower threshold to 0.25 for better detection, include all vehicle types
                if conf > 0.25 and label in ('car', 'bus', 'truck', 'motorbike', 'motorcycle', 'bicycle', 'bike', 'ambulance', 'firetruck', 'fire truck', 'police car', 'police'):
                    box = detection[0:4] * np.array([w, h, w, h])
                    (cX, cY, bw, bh) = box.astype('int')
                    x = int(cX - (bw / 2))
                    y = int(cY - (bh / 2))
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(conf)
                    labels.append(label)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, detector.conf_threshold, detector.nms_threshold)
        sel = idxs.flatten().tolist() if hasattr(idxs, 'flatten') else list(idxs) if isinstance(idxs, (list, tuple)) else []
        
        # Draw detections with enhanced visualization
        for idx, i in enumerate(sel, 1):
            x, y, bw, bh = boxes[i]
            conf = confidences[i]
            lbl = labels[i]
            
            # Update counts
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
            
            # Categorize vehicle type
            vtype = categorize_vehicle_type(lbl)
            if vtype in vehicle_type_counts:
                vehicle_type_counts[vtype] += 1
            
            # Draw bright green rectangle
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            
            # Prepare label text with confidence
            label_text = f"{lbl} #{class_counts[lbl]}"
            conf_text = f"{int(conf * 100)}%"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            (conf_width, conf_height), _ = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw filled rectangle background for text
            cv2.rectangle(img, 
                         (x, y - text_height - 8), 
                         (x + text_width + 5, y), 
                         (0, 255, 0), -1)
            
            # Draw label text in black on green background
            cv2.putText(img, label_text, 
                       (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw confidence below the box
            cv2.putText(img, conf_text, 
                       (x, y + bh + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        count = sum(class_counts.values())
    else:
        # Improved contour detection with better vehicle identification
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Use bilateral filter to preserve edges while reducing noise
        blur = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply Canny edge detection to find vehicle edges
        edges = cv2.Canny(blur, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Also use adaptive threshold for regions
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine edge and threshold results
        combined = cv2.bitwise_or(dilated, th)
        
        # Close gaps
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel2, iterations=2)
        
        # Fill holes
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filled = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel3, iterations=2)
        
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on aspect ratio and area
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            
            x, y, ww, hh = cv2.boundingRect(c)
            
            # Check aspect ratio - vehicles are typically wider than tall or roughly square
            aspect_ratio = ww / float(hh) if hh > 0 else 0
            
            # Filter out extremely thin or tall objects (likely not vehicles)
            if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                continue
            
            # Check if area is reasonable for a vehicle
            if area > (h * w * 0.5):  # Skip if object takes up more than 50% of image
                continue
            
            boxes.append((x, y, ww, hh, area))
            boxes.append((x, y, ww, hh, area))
        
        # Sort boxes by area (largest first) to prioritize larger vehicles
        boxes.sort(key=lambda b: b[4], reverse=True)
        
        # Remove duplicate/overlapping boxes using non-maximum suppression
        final_boxes = []
        for i, (x1, y1, w1, h1, a1) in enumerate(boxes):
            keep = True
            for j, (x2, y2, w2, h2, a2) in enumerate(final_boxes):
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    iou = intersection_area / float(min(a1, a2))
                    
                    # If significant overlap, skip this box
                    if iou > 0.5:
                        keep = False
                        break
            
            if keep:
                final_boxes.append((x1, y1, w1, h1, a1))
        
        # Draw detections with numbering
        for idx, (x, y, ww, hh, _) in enumerate(final_boxes, 1):
            # Draw bright green rectangle
            cv2.rectangle(img, (x, y), (x + ww, y + hh), (0, 255, 0), 3)
            
            # Add vehicle label with number
            label_text = f"Vehicle #{idx}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw filled rectangle background for text
            cv2.rectangle(img, 
                         (x, y - text_height - 8), 
                         (x + text_width + 5, y), 
                         (0, 255, 0), -1)
            
            # Draw label text in black on green background
            cv2.putText(img, label_text, 
                       (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        count = len(final_boxes)
        class_counts = {'car': count}
        vehicle_type_counts = {'car': count, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0}

    # Calculate metrics
    occupancy_rate = calculate_lane_occupancy(vehicle_type_counts, lane_pixels)
    queue_length = estimate_queue_length(count, vehicle_type_counts)
    
    # Add summary overlay on the image
    h_img, w_img = img.shape[:2]
    
    # Create semi-transparent overlay for summary
    overlay = img.copy()
    summary_height = 140
    cv2.rectangle(overlay, (10, 10), (350, summary_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw summary text
    y_offset = 35
    cv2.putText(img, f"VEHICLE DETECTION SUMMARY", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    y_offset += 25
    cv2.putText(img, f"Total Vehicles: {count}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display vehicle type breakdown
    y_offset += 20
    if vehicle_type_counts['car'] > 0:
        cv2.putText(img, f"Cars: {vehicle_type_counts['car']}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += 18
    
    if vehicle_type_counts['truck'] > 0:
        cv2.putText(img, f"Trucks: {vehicle_type_counts['truck']}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += 18
    
    if vehicle_type_counts['bus'] > 0:
        cv2.putText(img, f"Buses: {vehicle_type_counts['bus']}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += 18
    
    if vehicle_type_counts['two_wheeler'] > 0:
        cv2.putText(img, f"Two-Wheelers: {vehicle_type_counts['two_wheeler']}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += 18
    
    if vehicle_type_counts['bicycle'] > 0:
        cv2.putText(img, f"Bicycles: {vehicle_type_counts['bicycle']}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    # Save processed image
    base = os.path.basename(in_path)
    out_name = f"processed_{base}"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, img)

    return out_name, count, class_counts, vehicle_type_counts, occupancy_rate, queue_length


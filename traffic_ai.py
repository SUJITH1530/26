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


class YoloDetector:
    """YOLO-based detector using OpenCV DNN. If YOLO files are missing,
    it will fall back to a simple Haar cascade (if available).

    Place YOLO files under `models/`:
      - models/yolov3.cfg
      - models/yolov3.weights
      - models/coco.names
    """

    def __init__(self, model_dir='models', conf_threshold=0.5, nms_threshold=0.4):
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
            # fallback to Haar cascade for cars if present
            try:
                self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
                if self.haar.empty():
                    self.use_haar = False
                else:
                    self.use_haar = True
            except Exception:
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
                    if confidence > self.conf_threshold and label in ('car', 'bus', 'truck', 'motorbike'):
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


def process_image(in_path, out_dir, min_area=600):
    """Simple image-based vehicle estimator for uploads.
    Uses a contour-based approach to find blobs likely to be vehicles,
    draws bounding boxes and saves processed image into `out_dir`.

    Returns: (processed_filename, count)
    """
    img = cv2.imread(in_path)
    if img is None:
        return None, 0, {}

    orig = img.copy()
    h, w = img.shape[:2]

    # Resize if huge
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Try YOLO first for better results
    class_counts = {}
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
                if conf > detector.conf_threshold and label in ('car', 'bus', 'truck', 'motorbike'):
                    box = detection[0:4] * np.array([w, h, w, h])
                    (cX, cY, bw, bh) = box.astype('int')
                    x = int(cX - (bw / 2))
                    y = int(cY - (bh / 2))
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(conf)
                    labels.append(label)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, detector.conf_threshold, detector.nms_threshold)
        sel = idxs.flatten().tolist() if hasattr(idxs, 'flatten') else list(idxs) if isinstance(idxs, (list, tuple)) else []
        for i in sel:
            x, y, bw, bh = boxes[i]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            lbl = labels[i]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
            cv2.putText(img, lbl, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        count = sum(class_counts.values())
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            boxes.append((x, y, ww, hh))
        for (x, y, ww, hh) in boxes:
            cv2.rectangle(img, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        count = len(boxes)
        class_counts = {'vehicle': count}

    # Save processed image
    base = os.path.basename(in_path)
    out_name = f"processed_{base}"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, img)

    return out_name, count, class_counts


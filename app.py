from flask import Flask, render_template, jsonify, request
from traffic_ai import calculate_green_time, DetectorThread, process_image, detect_emergency_vehicles
from gpio_controller import GPIOController
import os
import time
import threading
import base64
import random
from werkzeug.utils import secure_filename
from datetime import datetime
from collections import deque


app = Flask(__name__, static_folder='static', template_folder='templates')

# Shared state updated by detector thread and uploads
shared_state = {
    'vehicles': 0,
    'green_time': calculate_green_time(0),
    'emergency_detected': False,
    'emergency_lane': None,
    'emergency_count': 0,
    'frames_without_emergency': 0,
    'emergency_threshold': 5,
    'manual_emergency_enabled': False,
    'manual_emergency_lane': None,
    'lanes': {
        'lane1': {
            'count': 0, 'green_time': 10, 'image': None, 
            'has_emergency': False, 'emergency_count': 0, 'frames_without_emergency': 0, 
            'vehicle_types': {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0},
            'occupancy_rate': 0.0,
            'queue_length': 0,
            'avg_speed': 0,  # km/h - placeholder for future implementation
            'traffic_history': deque(maxlen=60)  # Last 60 measurements (1 per second if polling)
        },
        'lane2': {
            'count': 0, 'green_time': 10, 'image': None, 
            'has_emergency': False, 'emergency_count': 0, 'frames_without_emergency': 0, 
            'vehicle_types': {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0},
            'occupancy_rate': 0.0,
            'queue_length': 0,
            'avg_speed': 0,
            'traffic_history': deque(maxlen=60)
        },
        'lane3': {
            'count': 0, 'green_time': 10, 'image': None, 
            'has_emergency': False, 'emergency_count': 0, 'frames_without_emergency': 0, 
            'vehicle_types': {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0},
            'occupancy_rate': 0.0,
            'queue_length': 0,
            'avg_speed': 0,
            'traffic_history': deque(maxlen=60)
        },
        'lane4': {
            'count': 0, 'green_time': 10, 'image': None, 
            'has_emergency': False, 'emergency_count': 0, 'frames_without_emergency': 0, 
            'vehicle_types': {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0},
            'occupancy_rate': 0.0,
            'queue_length': 0,
            'avg_speed': 0,
            'traffic_history': deque(maxlen=60)
        },
    },
    'peak_hours': [],  # Will store peak traffic times
    'off_peak_hours': [],  # Will store off-peak traffic times
}


def _update_lane(lane: str, count: int, processed_name: str | None, has_emergency: bool = False, 
                 emergency_count: int = 0, vehicle_types: dict = None, occupancy_rate: float = 0.0, 
                 queue_length: float = 0, avg_speed: int = 0):
    lane = lane if lane in shared_state['lanes'] else 'lane1'
    g = calculate_green_time(count)
    shared_state['lanes'][lane]['count'] = int(count)
    shared_state['lanes'][lane]['green_time'] = int(g)
    shared_state['lanes'][lane]['has_emergency'] = has_emergency
    shared_state['lanes'][lane]['emergency_count'] = int(emergency_count)
    shared_state['lanes'][lane]['occupancy_rate'] = float(occupancy_rate)
    shared_state['lanes'][lane]['queue_length'] = float(queue_length)
    shared_state['lanes'][lane]['avg_speed'] = int(avg_speed)
    
    # Update vehicle type counts
    if vehicle_types:
        shared_state['lanes'][lane]['vehicle_types'] = {
            'car': int(vehicle_types.get('car', 0)),
            'truck': int(vehicle_types.get('truck', 0)),
            'bus': int(vehicle_types.get('bus', 0)),
            'two_wheeler': int(vehicle_types.get('two_wheeler', 0)),
            'bicycle': int(vehicle_types.get('bicycle', 0))
        }
    
    # Add to traffic history with timestamp
    timestamp = datetime.now().strftime('%H:%M:%S')
    shared_state['lanes'][lane]['traffic_history'].append({
        'time': timestamp,
        'count': count,
        'occupancy': occupancy_rate
    })
    
    # Track consecutive frames without emergency detection
    if has_emergency and emergency_count > 0:
        # Emergency detected - reset frame counter and activate emergency mode
        shared_state['emergency_detected'] = True
        shared_state['emergency_lane'] = lane
        shared_state['emergency_count'] = int(emergency_count)
        shared_state['lanes'][lane]['frames_without_emergency'] = 0
        shared_state['frames_without_emergency'] = 0
    else:
        # No emergency in this frame
        shared_state['lanes'][lane]['frames_without_emergency'] += 1
        if shared_state['lanes'][lane]['frames_without_emergency'] >= shared_state['emergency_threshold']:
            # Vehicle has cleared this lane
            if shared_state.get('emergency_lane') == lane:
                shared_state['emergency_detected'] = False
                shared_state['emergency_lane'] = None
                shared_state['emergency_count'] = 0
                shared_state['frames_without_emergency'] = 0
    
    if processed_name:
        shared_state['lanes'][lane]['image'] = f"/static/processed/{processed_name}"

@app.route('/data')
def data():
    return jsonify(shared_state)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    response = app.make_response(render_template('dashboard.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/clear-images', methods=['POST'])
def clear_images():
    """Clear all stored lane preview images on dashboard load."""
    for lane in shared_state['lanes']:
        shared_state['lanes'][lane]['image'] = None
    return jsonify({'status': 'cleared'})


@app.route('/clear-lane', methods=['POST'])
def clear_lane():
    """Reset a single lane: clears image and zeroes counts."""
    data = request.get_json(silent=True) or {}
    lane = data.get('lane')
    if lane not in shared_state['lanes']:
        return jsonify({'error': 'invalid lane'}), 400
    shared_state['lanes'][lane]['image'] = None
    shared_state['lanes'][lane]['count'] = 0
    shared_state['lanes'][lane]['green_time'] = calculate_green_time(0)
    shared_state['lanes'][lane]['vehicle_types'] = {'car': 0, 'truck': 0, 'bus': 0, 'two_wheeler': 0, 'bicycle': 0}
    shared_state['lanes'][lane]['occupancy_rate'] = 0.0
    shared_state['lanes'][lane]['queue_length'] = 0
    shared_state['lanes'][lane]['avg_speed'] = 0
    shared_state['lanes'][lane]['traffic_history'].clear()
    return jsonify({'status': 'cleared', 'lanes': shared_state['lanes']})


@app.route('/activate-manual-emergency', methods=['POST'])
def activate_manual_emergency():
    """Activate manual emergency mode for a specific lane (highest priority override)."""
    data = request.get_json(silent=True) or {}
    lane = data.get('lane')
    if lane not in shared_state['lanes']:
        return jsonify({'error': 'invalid lane'}), 400
    
    shared_state['manual_emergency_enabled'] = True
    shared_state['manual_emergency_lane'] = lane
    print(f"🚨 MANUAL EMERGENCY ACTIVATED: {lane}")
    
    return jsonify({
        'status': 'manual emergency activated',
        'lane': lane,
        'manual_emergency_enabled': True,
        'manual_emergency_lane': lane
    })


@app.route('/clear-manual-emergency', methods=['POST'])
def clear_manual_emergency():
    """Clear manual emergency mode (return to normal traffic cycling)."""
    shared_state['manual_emergency_enabled'] = False
    shared_state['manual_emergency_lane'] = None
    print("✅ Manual emergency cleared - resuming normal traffic cycling")
    
    return jsonify({
        'status': 'manual emergency cleared',
        'manual_emergency_enabled': False,
        'manual_emergency_lane': None
    })


@app.route('/upload', methods=['POST'])
def upload_image():
    # Accept multipart file or JSON {image: dataURL}
    upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    processed_dir = os.path.join(os.path.dirname(__file__), 'static', 'processed')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    lane = request.form.get('lane') or (request.args.get('lane') or None)
    if 'image' in request.files:
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error': 'empty filename'}), 400
        filename = secure_filename(f.filename)
        in_path = os.path.join(upload_dir, filename)
        f.save(in_path)
    else:
        payload = request.get_json(silent=True) or {}
        data_url = payload.get('image')
        if not data_url:
            return jsonify({'error': 'no image provided'}), 400
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        raw = base64.b64decode(data_url)
        filename = f"capture_{str(os.getpid())}_{threading.get_ident()}.jpg"
        in_path = os.path.join(upload_dir, filename)
        with open(in_path, 'wb') as fh:
            fh.write(raw)

    processed_name, vehicles, classes, vehicle_types, occupancy_rate, queue_length = process_image(in_path, processed_dir)
    green = calculate_green_time(vehicles)
    has_emergency, emergency_count, emergency_details, emergency_conf = detect_emergency_vehicles(classes)
    
    # Placeholder for speed - would need video/consecutive frames
    avg_speed = 0  # km/h
    
    if lane:
        _update_lane(lane, vehicles, processed_name, has_emergency, emergency_count, 
                    vehicle_types, occupancy_rate, queue_length, avg_speed)
    return jsonify({
        'processed_image': f"/static/processed/{processed_name}",
        'vehicles': vehicles,
        'green_time': green,
        'classes': classes,
        'vehicle_types': vehicle_types,
        'occupancy_rate': occupancy_rate,
        'queue_length': queue_length,
        'avg_speed': avg_speed,
        'has_emergency': has_emergency,
        'emergency_count': emergency_count,
        'emergency_details': emergency_details,
        'emergency_conf': emergency_conf,
        'lanes': shared_state['lanes']
    })


@app.route('/upload-multi', methods=['POST'])
def upload_multi():
    """Accept up to four images for lanes lane1..lane4 in one request.
    Multipart fields should be 'lane1', 'lane2', 'lane3', 'lane4'.
    Alternatively, JSON body with { images: { lane1: dataURL, ... } }.
    Returns consolidated lane states.
    """
    upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    processed_dir = os.path.join(os.path.dirname(__file__), 'static', 'processed')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    results = {}
    if request.files:
        for lane in ('lane1', 'lane2', 'lane3', 'lane4'):
            f = request.files.get(lane)
            if not f:
                continue
            filename = secure_filename(f.filename or f"{lane}.jpg")
            in_path = os.path.join(upload_dir, filename)
            f.save(in_path)
            processed_name, vehicles, classes, vehicle_types, occupancy_rate, queue_length = process_image(in_path, processed_dir)
            has_emergency, emergency_count, emergency_details, emergency_conf = detect_emergency_vehicles(classes)
            avg_speed = 0
            _update_lane(lane, vehicles, processed_name, has_emergency, emergency_count, 
                        vehicle_types, occupancy_rate, queue_length, avg_speed)
            results[lane] = {
                'processed_image': f"/static/processed/{processed_name}",
                'vehicles': vehicles,
                'green_time': calculate_green_time(vehicles),
                'classes': classes,
                'vehicle_types': vehicle_types,
                'occupancy_rate': occupancy_rate,
                'queue_length': queue_length,
                'avg_speed': avg_speed,
                'has_emergency': has_emergency,
                'emergency_count': emergency_count,
                'emergency_details': emergency_details,
                'emergency_conf': emergency_conf,
            }
    else:
        data = request.get_json(silent=True) or {}
        images = (data.get('images') or {}) if isinstance(data, dict) else {}
        for lane, img_b64 in images.items():
            if lane not in ('lane1', 'lane2', 'lane3', 'lane4'):
                continue
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]
            raw = base64.b64decode(img_b64)
            filename = f"{lane}_{random.randint(1000,9999)}.jpg"
            in_path = os.path.join(upload_dir, filename)
            with open(in_path, 'wb') as fh:
                fh.write(raw)
            processed_name, vehicles, classes, vehicle_types, occupancy_rate, queue_length = process_image(in_path, processed_dir)
            has_emergency, emergency_count, emergency_details, emergency_conf = detect_emergency_vehicles(classes)
            avg_speed = 0
            _update_lane(lane, vehicles, processed_name, has_emergency, emergency_count, 
                        vehicle_types, occupancy_rate, queue_length, avg_speed)
            results[lane] = {
                'processed_image': f"/static/processed/{processed_name}",
                'vehicles': vehicles,
                'green_time': calculate_green_time(vehicles),
                'classes': classes,
                'vehicle_types': vehicle_types,
                'occupancy_rate': occupancy_rate,
                'queue_length': queue_length,
                'avg_speed': avg_speed,
                'has_emergency': has_emergency,
                'emergency_count': emergency_count,
                'emergency_details': emergency_details,
                'emergency_conf': emergency_conf,
            }

    # Build order by highest count
    order = sorted(shared_state['lanes'].items(), key=lambda kv: kv[1]['count'], reverse=True)
    order = [k for k, _ in order]
    return jsonify({'lanes': shared_state['lanes'], 'results': results, 'order': order})


def start_detector_and_gpio():
    # detector source can be set with VIDEO_SOURCE env var; skip if none
    source = os.environ.get('VIDEO_SOURCE', 'none')
    model_dir = os.environ.get('MODEL_DIR', 'models')
    disable_detector = os.environ.get('DISABLE_DETECTOR', '').lower() in ('1', 'true', 'yes')

    detector = None
    if not disable_detector and str(source).lower() not in ('', 'none', 'off'):
        detector = DetectorThread(shared_state, source=source, model_dir=model_dir)
        detector.start()
    else:
        print("Detector disabled (no camera/video source provided)")

    # Start a simple thread to control GPIO lights according to green_time
    gpio = GPIOController()

    def gpio_loop():
        try:
            while True:
                # PRIORITY 1: Manual emergency override (highest priority - traffic police control)
                if shared_state.get('manual_emergency_enabled'):
                    manual_lane = shared_state.get('manual_emergency_lane')
                    print(f"🚨 MANUAL EMERGENCY: {manual_lane} - GREEN INDEFINITELY (no timer)")
                    gpio.set_light('green')
                    time.sleep(0.5)
                
                # PRIORITY 2: AI-detected emergency vehicles (automatic detection)
                elif shared_state.get('emergency_detected'):
                    emergency_lane = shared_state.get('emergency_lane')
                    emergency_count = shared_state.get('emergency_count', 0)
                    print(f"🚨 EMERGENCY ACTIVE: {emergency_lane} has {emergency_count} vehicle(s) - GREEN UNTIL VEHICLE CROSSES")
                    gpio.set_light('green')
                    time.sleep(0.5)
                
                # PRIORITY 3: Normal traffic cycling (density-based)
                else:
                    g = shared_state.get('green_time', 10)
                    # simple cycle: green -> amber -> red
                    gpio.cycle_green(g)
                    gpio.set_light('amber')
                    # amber short
                    time.sleep(2)
                    gpio.set_light('red')
                    # red for a constant small interval
                    time.sleep(3)
        except Exception:
            gpio.cleanup()

    t = threading.Thread(target=gpio_loop, daemon=True)
    t.start()

    return detector, gpio


if __name__ == "__main__":
    detector, gpio = start_detector_and_gpio()
    port = int(os.environ.get("PORT", 5000))
    try:
        app.run(debug=False, host="0.0.0.0", port=port)
    finally:
        if detector:
            detector.stop()
        gpio.cleanup()

from flask import Flask, render_template, jsonify, request
from traffic_ai import calculate_green_time, DetectorThread, process_image
from gpio_controller import GPIOController
import os
import threading
import base64
from werkzeug.utils import secure_filename


app = Flask(__name__, static_folder='static', template_folder='templates')

# Shared state updated by detector thread and uploads
shared_state = {
    'vehicles': 0,
    'green_time': calculate_green_time(0),
    'lanes': {
        'lane1': {'count': 0, 'green_time': 10, 'image': None},
        'lane2': {'count': 0, 'green_time': 10, 'image': None},
        'lane3': {'count': 0, 'green_time': 10, 'image': None},
        'lane4': {'count': 0, 'green_time': 10, 'image': None},
    },
}


def _update_lane(lane: str, count: int, processed_name: str | None):
    lane = lane if lane in shared_state['lanes'] else 'lane1'
    g = calculate_green_time(count)
    shared_state['lanes'][lane]['count'] = int(count)
    shared_state['lanes'][lane]['green_time'] = int(g)
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
    return render_template('dashboard.html')


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
    return jsonify({'status': 'cleared', 'lanes': shared_state['lanes']})


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

    processed_name, vehicles, classes = process_image(in_path, processed_dir)
    green = calculate_green_time(vehicles)
    if lane:
        _update_lane(lane, vehicles, processed_name)
    return jsonify({
        'processed_image': f"/static/processed/{processed_name}",
        'vehicles': vehicles,
        'green_time': green,
        'classes': classes,
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
            processed_name, vehicles, classes = process_image(in_path, processed_dir)
            _update_lane(lane, vehicles, processed_name)
            results[lane] = {
                'processed_image': f"/static/processed/{processed_name}",
                'vehicles': vehicles,
                'green_time': calculate_green_time(vehicles),
                'classes': classes,
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
            processed_name, vehicles, classes = process_image(in_path, processed_dir)
            _update_lane(lane, vehicles, processed_name)
            results[lane] = {
                'processed_image': f"/static/processed/{processed_name}",
                'vehicles': vehicles,
                'green_time': calculate_green_time(vehicles),
                'classes': classes,
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
                g = shared_state.get('green_time', 10)
                # simple cycle: green -> amber -> red
                gpio.cycle_green(g)
                gpio.set_light('amber')
                # amber short
                import time
                time.sleep(2)
                gpio.set_light('red')
                # red for a constant small interval (can be extended)
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

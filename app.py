import os
import uuid
import logging
import io
import time
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import threading

from ultralytics import YOLO
from rvt.default import DefaultValues
from flask import (
    Flask, request, render_template, send_from_directory,
    Response, stream_with_context, jsonify
)
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

ALLOWED_EXTENSIONS = {'asc', 'tif', 'tiff'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = 'some_random_secret_key'

MODEL_PATH        = 'best.pt'
PATCH_SIZE        = 200
SLRM_RAD_CELL     = 50
VE_FACTOR         = 5
MAX_SLOPE_PERCENT = 1.55
PROXIMITY_M       = 5.2
QUANTILE_PCT      = 8.33

PROGRESS = {}
RESULTS  = {}

model = YOLO(MODEL_PATH)

def get_slice(img, cell_size, max_slope=MAX_SLOPE_PERCENT):
    def line_2p(p1, p2, x):
        return (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1]

    def slope(slce):
        p1 = [len(slce)//4, np.mean(slce[:len(slce)//2])]
        p2 = [len(slce)*3//4, np.mean(slce[len(slce)//2:])]
        x = np.linspace(0, len(slce), 10) * cell_size
        y = line_2p(p1, p2, x)
        return x, y

    def get_trend(slce):
        x, y = slope(slce)
        dx, dy = x.max() - x.min(), y.max() - y.min()
        length = np.hypot(dx, dy)
        slp = dy * 100.0 / length
        return slp < max_slope

    dirs = {
        'h':    img[img.shape[0]//2, :],
        'v':    img[:, img.shape[1]//2],
        'dlr':  np.diag(img),
        'drl':  np.diag(np.fliplr(img)),
    }
    return all(get_trend(slc) for slc in dirs.values())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_ascii_grid(path):
    header = {}
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(6):
            k, v = f.readline().split()
            header[k.lower()] = float(v)
        arr = np.loadtxt(f)
    return header, arr

def extract_patches(img, out_dir, sid):
    os.makedirs(out_dir, exist_ok=True)
    nrows, ncols = img.shape
    coords = []
    half = PATCH_SIZE // 2

    for r in range(0, nrows - PATCH_SIZE + 1, PATCH_SIZE):
        for c in range(0, ncols - PATCH_SIZE + 1, PATCH_SIZE):
            coords.append((r, c))

    for r in range(0, nrows - PATCH_SIZE - half + 1, PATCH_SIZE):
        for c in range(0, ncols - PATCH_SIZE - half + 1, PATCH_SIZE):
            coords.append((r + half, c + half))

    total = len(coords)

    def save_rgb(patch, fn):
        rgb = np.stack([patch]*3, -1)
        plt.imsave(fn, rgb, vmin=0, vmax=1)

    for i, (r, c) in enumerate(coords, start=1):
        patch = img[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
        fn = os.path.join(out_dir, f'patch_{r}_{c}_{PATCH_SIZE}.png')
        save_rgb(patch, fn)
        pct = math.floor(i * 50 / total)
        PROGRESS[sid] = pct

    PROGRESS[sid] = 50
    return len(coords)

def process_dem(path, sid):
    frames = []

    def capture(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    PROGRESS[sid] = 0

    header, arr = read_ascii_grid(path)
    nrows = int(header['nrows'])
    ndv   = header.get('nodata_value', None)
    cellsize = header['cellsize']

    xll = header.get('xllcorner', header.get('xllcenter', 0))
    yll = header.get('yllcorner', header.get('yllcenter', 0))

    dv = DefaultValues()
    dv.ve_factor     = VE_FACTOR
    dv.slrm_rad_cell = SLRM_RAD_CELL
    slrm = dv.get_slrm(arr, ndv)
    lo, hi = np.percentile(slrm[~np.isnan(slrm)], [2, 98])
    stretched = np.clip(slrm, lo, hi)
    stretched = (stretched - lo) / (hi - lo)

    # Visualization steps
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(stretched, cmap='gray', origin='upper')
    ax.set_title('1. SLRM-растяжение')
    capture(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(stretched, cmap='gray', origin='upper')
    for r in range(0, nrows, PATCH_SIZE):
        for c in range(0, arr.shape[1], PATCH_SIZE):
            ax.add_patch(plt.Rectangle((c, r), PATCH_SIZE, PATCH_SIZE,
                                       edgecolor='cyan', facecolor='none', linewidth=0.5))
    ax.set_title('2. Разбиение на патчи')
    capture(fig)
    plt.close(fig)

    # Patch extraction and detection
    patch_dir = os.path.join(app.config['STATIC_FOLDER'], f'patches_{sid}')
    total_patches = extract_patches(stretched, patch_dir, sid)

    detected = []
    stream = model.predict(
        source=patch_dir,
        conf=0.5,
        save=False,
        imgsz=PATCH_SIZE,
        stream=True
    )
    for i, res in enumerate(stream, start=1):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        img = plt.imread(res.path)
        ax.imshow(img)
        name = os.path.basename(res.path)
        _, r_str, c_str, _ = name.split('_')
        r0, c0 = int(r_str), int(c_str)
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       edgecolor='red', linewidth=1.5))
            cx = (x1 + x2)/2 + c0
            cy = (y1 + y2)/2 + r0
            detected.append((cx, cy))
        ax.axis('off')
        ax.set_title(f'3. Детекция {name}')
        capture(fig)
        plt.close(fig)

        pct = 50 + math.floor(i * 49 / total_patches)
        PROGRESS[sid] = min(pct, 99)

    # Filtering by elevation
    valid = arr[arr != ndv] if ndv is not None else arr[~np.isnan(arr)]
    _, high_q = np.percentile(valid, [0, QUANTILE_PCT])
    quant_filtered = []
    for x, y in detected:
        ri, ci = int(round(y)), int(round(x))
        if 0 <= ri < arr.shape[0] and 0 <= ci < arr.shape[1]:
            if arr[ri, ci] >= high_q:
                quant_filtered.append((x, y))

    # Trend filtering
    confirmed = []
    half = PATCH_SIZE // 2
    for x, y in quant_filtered:
        ri, ci = int(round(y)), int(round(x))
        if ri-half < 0 or ri+half > arr.shape[0] or ci-half < 0 or ci+half > arr.shape[1]:
            continue
        patch = arr[ri-half:ri+half, ci-half:ci+half]
        if get_slice(patch, cellsize):
            confirmed.append((x, y))

    # Proximity deduplication
    thresh_px = PROXIMITY_M / cellsize
    unique_pts = []
    for pt in confirmed:
        if not any(math.hypot(pt[0]-u[0], pt[1]-u[1]) < thresh_px for u in unique_pts):
            unique_pts.append(pt)
    final_pts = unique_pts

    # Save map image
    png_name = f'map_{sid}.png'
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.imshow(stretched, cmap='gray', origin='upper')
    if final_pts:
        xs, ys = zip(*final_pts)
        ax.plot(xs, ys, 'rx', markersize=6)
    ax.set_title('Обнаруженные курганы (после фильтрации)')
    fig.savefig(os.path.join(app.config['STATIC_FOLDER'], png_name), dpi=100)
    plt.close(fig)

    # Save coordinates in absolute projection
    csv_name = f'coords_{sid}.csv'
    with open(os.path.join(app.config['STATIC_FOLDER'], csv_name), 'w', encoding='utf-8') as f:
        f.write('Name;X;Y\n')
        for idx, (x, y) in enumerate(final_pts, start=1):
            # Convert pixel to map coordinates
            x_abs = xll + x * cellsize
            y_abs = yll + (nrows - y) * cellsize
            name = f'Candidate_{idx}'
            f.write(f'{name};{x_abs:.3f};{y_abs:.3f}\n')

    # Save animation
    anim_name = f'animation_{sid}.gif'
    anim_path = os.path.join(app.config['STATIC_FOLDER'], anim_name)
    with imageio.get_writer(anim_path, mode='I', fps=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    RESULTS[sid] = {'png': png_name, 'csv': csv_name, 'anim': anim_name}
    PROGRESS[sid] = 100
    logging.info(f'[{sid}] Обработка завершена.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_json', methods=['POST'])
def upload_json():
    file = request.files.get('demfile')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    fn = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    file.save(path)
    sid = str(uuid.uuid4())
    PROGRESS[sid] = 0
    threading.Thread(target=process_dem, args=(path, sid), daemon=True).start()
    return jsonify({'sid': sid})

@app.route('/stream/<sid>')
def stream(sid):
    if sid not in PROGRESS:
        return 'Session not found', 404

    def event_stream():
        while True:
            p = PROGRESS.get(sid, 0)
            message = {'percent': p}
            if p >= 98 and p < 100:
                message['note'] = 'Подождите пожалуйста, обрабатываем кандидатов и рисуем карту для вас'
            yield f"data: {json.dumps(message)}\n\n"
            if p >= 100:
                break
            time.sleep(0.3)

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

@app.route('/result/<sid>')
def show_result(sid):
    res = RESULTS.get(sid)
    if not res:
        return 'Result not ready', 404
    return render_template(
        'result.html', image_file=res['png'], csv_file=res['csv'], anim_file=res['anim']
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

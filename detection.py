import os
import logging
import io
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from rvt.default import DefaultValues
from ultralytics import YOLO
from config import Config


class Detection:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)
        self.progress = {}
        self.results = {}

    def process_dem(self, path, sid, folder_path):
        frames = []

        def capture(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

        self.progress[sid] = 0

        header, arr = self._read_ascii_grid(path)
        nrows = int(header['nrows'])
        ndv = header.get('nodata_value', None)
        cellsize = header['cellsize']

        xll = header.get('xllcorner', header.get('xllcenter', 0))
        yll = header.get('yllcorner', header.get('yllcenter', 0))

        dv = DefaultValues()
        dv.ve_factor = Config.VE_FACTOR
        dv.slrm_rad_cell = Config.SLRM_RAD_CELL
        slrm = dv.get_slrm(arr, ndv)
        lo, hi = np.percentile(slrm[~np.isnan(slrm)], [2, 98])
        stretched = np.clip(slrm, lo, hi)
        stretched = (stretched - lo) / (hi - lo)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(stretched, cmap='gray', origin='upper')
        ax.set_title('1. SLRM-растяжение')
        capture(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(stretched, cmap='gray', origin='upper')
        for r in range(0, nrows, Config.PATCH_SIZE):
            for c in range(0, arr.shape[1], Config.PATCH_SIZE):
                ax.add_patch(plt.Rectangle(
                    (c, r), Config.PATCH_SIZE, Config.PATCH_SIZE,
                    edgecolor='cyan', facecolor='none', linewidth=0.5
                ))
        ax.set_title('2. Разбиение на патчи')
        capture(fig)
        plt.close(fig)

        patch_dir = os.path.join(folder_path, f'patches_{sid}')
        total_patches = self._extract_patches(stretched, patch_dir, sid)

        detected = []
        stream = self.model.predict(
            source=patch_dir,
            conf=0.5,
            save=False,
            imgsz=Config.PATCH_SIZE,
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
                ax.add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    edgecolor='red', linewidth=1.5
                ))
                cx = (x1 + x2) / 2 + c0
                cy = (y1 + y2) / 2 + r0
                detected.append((cx, cy))
            ax.axis('off')
            ax.set_title(f'3. Детекция {name}')
            capture(fig)
            plt.close(fig)

            pct = 50 + math.floor(i * 49 / total_patches)
            self.progress[sid] = min(pct, 99)

        valid = arr[arr != ndv] if ndv is not None else arr[~np.isnan(arr)]
        _, high_q = np.percentile(valid, [0, Config.QUANTILE_PCT])
        quant_filtered = []
        for x, y in detected:
            ri, ci = int(round(y)), int(round(x))
            if 0 <= ri < arr.shape[0] and 0 <= ci < arr.shape[1]:
                if arr[ri, ci] >= high_q:
                    quant_filtered.append((x, y))

        confirmed = []
        half = Config.PATCH_SIZE // 2
        for x, y in quant_filtered:
            ri, ci = int(round(y)), int(round(x))
            if ri - half < 0 or ri + half > arr.shape[0] or ci - half < 0 or ci + half > arr.shape[1]:
                continue
            patch = arr[ri - half:ri + half, ci - half:ci + half]
            if self._get_slice(patch, cellsize):
                confirmed.append((x, y))

        thresh_px = Config.PROXIMITY_M / cellsize
        unique_pts = []
        for pt in confirmed:
            if not any(math.hypot(pt[0] - u[0], pt[1] - u[1]) < thresh_px for u in unique_pts):
                unique_pts.append(pt)
        final_pts = unique_pts

        png_name = f'map_{sid}.png'
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(stretched, cmap='gray', origin='upper')
        if final_pts:
            xs, ys = zip(*final_pts)
            ax.plot(xs, ys, 'rx', markersize=6)
        ax.set_title('Обнаруженные курганы (после фильтрации)')
        fig.savefig(os.path.join(folder_path, png_name), dpi=100)
        plt.close(fig)

        csv_name = f'coords_{sid}.csv'
        with open(os.path.join(folder_path, csv_name), 'w', encoding='utf-8') as f:
            f.write('Name;X;Y\n')
            for idx, (x, y) in enumerate(final_pts, start=1):
                x_abs = xll + x * cellsize
                y_abs = yll + (nrows - y) * cellsize
                name = f'Candidate_{idx}'
                f.write(f'{name};{x_abs:.3f};{y_abs:.3f}\n')

        anim_name = f'animation_{sid}.gif'
        anim_path = os.path.join(folder_path, anim_name)
        with imageio.get_writer(anim_path, mode='I', fps=1) as writer:
            for frame in frames:
                writer.append_data(frame)

        self.results[sid] = {'png': png_name, 'csv': csv_name, 'anim': anim_name}
        self.progress[sid] = 100
        logging.info(f'[{sid}] Обработка завершена.')

    def _get_slice(self, img, cell_size, max_slope=Config.MAX_SLOPE_PERCENT):
        def line_2p(p1, p2, x):
            return (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1]

        def slope(slce):
            p1 = [len(slce) // 4, np.mean(slce[:len(slce) // 2])]
            p2 = [len(slce) * 3 // 4, np.mean(slce[len(slce) // 2:])]
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
            'h': img[img.shape[0] // 2, :],
            'v': img[:, img.shape[1] // 2],
            'dlr': np.diag(img),
            'drl': np.diag(np.fliplr(img)),
        }

        return all(get_trend(slc) for slc in dirs.values())

    def _read_ascii_grid(self, path):
        header = {}
        with open(path, 'r', encoding='utf-8') as f:
            for _ in range(6):
                k, v = f.readline().split()
                header[k.lower()] = float(v)
            arr = np.loadtxt(f)

        return header, arr

    def _extract_patches(self, img, out_dir, sid):
        os.makedirs(out_dir, exist_ok=True)
        nrows, ncols = img.shape
        coords = []
        half = Config.PATCH_SIZE // 2

        for r in range(0, nrows - Config.PATCH_SIZE + 1, Config.PATCH_SIZE):
            for c in range(0, ncols - Config.PATCH_SIZE + 1, Config.PATCH_SIZE):
                coords.append((r, c))

        for r in range(0, nrows - Config.PATCH_SIZE - half + 1, Config.PATCH_SIZE):
            for c in range(0, ncols - Config.PATCH_SIZE - half + 1, Config.PATCH_SIZE):
                coords.append((r + half, c + half))

        total = len(coords)

        def save_rgb(patch, fn):
            rgb = np.stack([patch] * 3, -1)
            plt.imsave(fn, rgb, vmin=0, vmax=1)

        for i, (r, c) in enumerate(coords, start=1):
            patch = img[r:r + Config.PATCH_SIZE, c:c + Config.PATCH_SIZE]
            fn = os.path.join(out_dir, f'patch_{r}_{c}_{Config.PATCH_SIZE}.png')
            save_rgb(patch, fn)
            pct = math.floor(i * 50 / total)
            self.progress[sid] = pct

        self.progress[sid] = 50

        return len(coords)
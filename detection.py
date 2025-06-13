import os
import io
import math
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2

from scipy.stats import skew
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern
from ultralytics import YOLO

from rvt.default import DefaultValues
from config import Config

matplotlib.use("Agg")


def get_slice(img, cell_size, max_slope=Config.MAX_SLOPE_PERCENT):
    def line_2p(p1, p2, x):
        return (
            (x - p1[0]) * (p2[1] - p1[1]) /
            (p2[0] - p1[0]) + p1[1]
        )

    def slope_line(slce):
        p1 = [len(slce) // 4, np.mean(slce[:len(slce)//2])]
        p2 = [
            len(slce) * 3 // 4,
            np.mean(slce[len(slce)//2:])
        ]
        x = np.linspace(0, len(slce), 10) * cell_size
        y = line_2p(p1, p2, x)
        return x, y

    def check_trend(slce):
        x, y = slope_line(slce)
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        length = np.hypot(dx, dy)
        slope_pct = dy * 100.0 / length
        return slope_pct < max_slope

    dirs = {
        "h": img[img.shape[0]//2, :],
        "v": img[:, img.shape[1]//2],
        "dlr": np.diag(img),
        "drl": np.diag(np.fliplr(img)),
    }

    return all(check_trend(s) for s in dirs.values())


def get_envelope(slce, window=10):
    padded = ([slce[0]] * window +
              list(slce) +
              [slce[-1]] * window)
    env = np.convolve(padded, np.ones(window), "same") / window
    return env[window:-window]


def get_slice_array(img, direction):
    if direction == "h":
        return img[img.shape[0]//2, :]
    if direction == "v":
        return img[:, img.shape[1]//2]
    if direction == "dlr":
        return img.diagonal()
    return np.fliplr(img).diagonal()


def remove_trend(slce, cell_size):
    p1 = [len(slce)//4, np.mean(slce[:len(slce)//2])]
    p2 = [len(slce)*3//4, np.mean(slce[len(slce)//2:])]
    x = np.linspace(0, len(slce), len(slce)) * cell_size
    trend = (
        (x - p1[0]) * (p2[1] - p1[1]) /
        (p2[0] - p1[0]) + p1[1]
    )
    return slce - trend


def filter_by_skew(
    img,
    cell_size,
    skew_low=Config.SKEW_LOW,
    skew_hi=Config.SKEW_HIGH,
):
    for d in ["h", "v", "dlr", "drl"]:
        slc = get_slice_array(img, d)
        detr = remove_trend(slc, cell_size)
        env = get_envelope(detr)
        if not (skew_low < skew(env) < skew_hi):
            return False
    return True


def slope_variance(
    img,
    cell_size,
    thresh=Config.VARIANCE_THRESH,
):
    vars_list = []
    for d in ["h", "v", "dlr", "drl"]:
        slc = get_slice_array(img, d)
        detr = remove_trend(slc, cell_size)
        vars_list.append(np.var(detr))
    return max(vars_list) <= thresh


def filter_by_circularity(
    patch,
    level=Config.CIRCULARITY_LEVEL,
    circ_thresh=Config.CIRCULARITY_THRESH,
):
    p = ((patch - patch.min()) /
         (patch.max() - patch.min() + 1e-6))
    _, bw = cv2.threshold(
        (p * 255).astype(np.uint8),
        int(level * 255),
        255,
        cv2.THRESH_BINARY,
    )
    cnts, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    circ = 4 * math.pi * area / (perim**2 + 1e-6)
    return circ > circ_thresh


def compute_curvatures(
    patch,
    cell_size,
):
    Z = patch
    Zxx = (
        Z[2:, 1:-1] - 2 * Z[1:-1, 1:-1] +
        Z[:-2, 1:-1]
    ) / (cell_size**2)
    Zyy = (
        Z[1:-1, 2:] - 2 * Z[1:-1, 1:-1] +
        Z[1:-1, :-2]
    ) / (cell_size**2)
    return Zxx + Zyy


def filter_by_curvature(
    patch,
    cell_size,
    H_thresh=Config.CURVATURE_THRESH,
):
    H = compute_curvatures(patch, cell_size)
    cy = H.shape[0]//2
    cx = H.shape[1]//2
    win = H[
        max(cy-11, 0):cy+12,
        max(cx-11, 0):cx+12,
    ]
    return win.mean() > H_thresh


def compute_tpi(dem, window):
    return dem - uniform_filter(
        dem, size=window, mode="nearest"
    )


def lbp_entropy(patch, P, R):
    norm = ((patch - patch.min()) /
            (patch.max() - patch.min() + 1e-6))
    img8 = (norm * 255).astype(np.uint8)
    lbp = local_binary_pattern(
        img8, P, R, method="uniform"
    )
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=P+2,
        range=(0, P+1),
        density=True,
    )
    return -np.sum(hist * np.log2(hist + 1e-6))


class Detection:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)
        self.progress = {}
        self.results = {}

    def process_dem(self, path, sid, folder_path):
        frames = []

        def capture(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

        self.progress[sid] = 0
        header, arr = self._read_ascii_grid(path)

        nrows = int(header["nrows"])
        ndv = header.get("nodata_value")
        cellsize = header["cellsize"]
        xll = header.get("xllcorner", 0)
        yll = header.get("yllcorner", 0)
        top = yll + nrows * cellsize

        dv = DefaultValues()
        dv.ve_factor = Config.VE_FACTOR
        dv.slrm_rad_cell = Config.SLRM_RAD_CELL
        slrm = dv.get_slrm(arr, ndv)
        lo, hi = np.percentile(
            slrm[~np.isnan(slrm)], [2, 98]
        )
        stretched = np.clip(slrm, lo, hi)
        stretched = (stretched - lo) / (hi - lo)

        fig, ax = plt.subplots(
            figsize=(6, 6),
            dpi=100,
        )
        ax.imshow(stretched, cmap="gray", origin="upper")
        ax.set_title("SLRM-растяжение")
        capture(fig)
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(6, 6),
            dpi=100,
        )
        ax.imshow(stretched, cmap="gray", origin="upper")
        for r in range(0, nrows, Config.PATCH_SIZE):
            for c in range(0, arr.shape[1], Config.PATCH_SIZE):
                ax.add_patch(
                    plt.Rectangle(
                        (c, r),
                        Config.PATCH_SIZE,
                        Config.PATCH_SIZE,
                        edgecolor="cyan",
                        facecolor="none",
                        linewidth=0.5,
                    )
                )
        ax.set_title("Разбиение на патчи")
        capture(fig)
        plt.close(fig)

        patch_dir = os.path.join(
            folder_path, f"patches_{sid}"
        )
        total_patches = self._extract_patches(
            stretched, patch_dir, sid
        )

        detected = []
        stream = self.model.predict(
            source=patch_dir,
            conf=0.5,
            save=False,
            imgsz=Config.PATCH_SIZE,
            stream=True,
        )
        for i, res in enumerate(stream, start=1):
            _, r_str, c_str, _ = os.path.basename(
                res.path
            ).split("_")
            r0 = int(r_str)
            c0 = int(c_str)
            for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                cx = (x1 + x2) / 2 + c0
                cy = (y1 + y2) / 2 + r0
                detected.append((cx, cy))

            self.progress[sid] = min(
                50 + math.floor(i * 49 / total_patches),
                99,
            )

        valid = (
            arr[arr != ndv]
            if ndv is not None else arr[~np.isnan(arr)]
        )
        low_q, high_q = np.percentile(
            valid, [0, Config.QUANTILE_PCT]
        )
        pts = [
            (cx, cy)
            for cx, cy in detected
            if arr[int(round(cy)), int(round(cx))] >= high_q
        ]

        half = Config.PATCH_SIZE // 2
        filtered = []

        for cx, cy in pts:
            ri = int(round(cy))
            ci = int(round(cx))
            patch = arr[
                ri-half:ri+half,
                ci-half:ci+half,
            ]
            if patch.shape != (
                Config.PATCH_SIZE,
                Config.PATCH_SIZE,
            ):
                continue
            if get_slice(patch, cellsize):
                filtered.append((cx, cy))

        pts = filtered
        filtered = []

        thr = Config.PROXIMITY_M / cellsize
        unique_pts = []
        for pt in pts:
            if not any(
                math.hypot(
                    pt[0]-u[0],
                    pt[1]-u[1],
                ) < thr
                for u in unique_pts
            ):
                unique_pts.append(pt)
        pts = unique_pts

        for cx, cy in pts:
            ri = int(round(cy))
            ci = int(round(cx))
            patch = arr[
                ri-half:ri+half,
                ci-half:ci+half,
            ]
            if patch.shape != (
                Config.PATCH_SIZE,
                Config.PATCH_SIZE,
            ):
                continue
            if filter_by_skew(patch, cellsize):
                filtered.append((cx, cy))
        pts = filtered
        filtered = []

        for cx, cy in pts:
            ri = int(round(cy))
            ci = int(round(cx))
            patch = arr[
                ri-half:ri+half,
                ci-half:ci+half,
            ]
            if patch.shape != (
                Config.PATCH_SIZE,
                Config.PATCH_SIZE,
            ):
                continue
            if slope_variance(patch, cellsize):
                filtered.append((cx, cy))
        pts = filtered
        filtered = []

        for cx, cy in pts:
            ri = int(round(cy))
            ci = int(round(cx))
            patch = arr[
                ri-half:ri+half,
                ci-half:ci+half,
            ]
            if patch.shape != (
                Config.PATCH_SIZE,
                Config.PATCH_SIZE,
            ):
                continue
            if filter_by_circularity(patch):
                filtered.append((cx, cy))
        pts = filtered
        filtered = []

        for cx, cy in pts:
            ri = int(round(cy))
            ci = int(round(cx))
            patch = arr[
                ri-half:ri+half,
                ci-half:ci+half,
            ]
            if patch.shape != (
                Config.PATCH_SIZE,
                Config.PATCH_SIZE,
            ):
                continue
            if filter_by_curvature(patch, cellsize):
                filtered.append((cx, cy))
        pts = filtered
        filtered = []

        tpi_map = compute_tpi(
            arr,
            Config.PATCH_SIZE *
            Config.TPI_WINDOW_MULTIPLIER,
        )
        tpis = [
            tpi_map[
                int(round(cy)),
                int(round(cx)),
            ]
            for cx, cy in pts
        ]
        tpi_thr = np.percentile(
            tpis, Config.TPI_PERCENTILE
        )
        for (cx, cy), val in zip(pts, tpis):
            if val >= tpi_thr:
                filtered.append((cx, cy))
        pts = filtered
        filtered = []

        entropies = [
            lbp_entropy(
                arr[
                    int(round(cy))-half:
                    int(round(cy))+half,
                    int(round(cx))-half:
                    int(round(cx))+half,
                ],
                Config.LBP_P,
                Config.LBP_R,
            )
            for cx, cy in pts
        ]
        ent_thr = np.percentile(
            entropies,
            Config.LBP_ENTROPY_PERCENTILE,
        )
        final_pts = [
            pt for pt, ent in zip(pts, entropies)
            if ent <= ent_thr
        ]

        png = f"map_{sid}.png"
        fig, ax = plt.subplots(
            figsize=(10, 8),
            dpi=100,
        )
        ax.imshow(stretched, cmap="gray", origin="upper")
        if final_pts:
            xs, ys = zip(*final_pts)
            ax.plot(xs, ys, "rx", markersize=6)
        ax.set_title("Обнаруженные объекты")
        fig.savefig(
            os.path.join(folder_path, png),
            dpi=100,
        )
        plt.close(fig)

        csv_name = f"coords_{sid}.csv"
        with open(
            os.path.join(folder_path, csv_name),
            "w", encoding="utf-8"
        ) as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Name", "X", "Y"])
            for idx, (cx, cy) in enumerate(final_pts, start=1):
                gx = xll + cx*cellsize
                gy = top - cy*cellsize
                writer.writerow(
                    [f"Candidate_{idx}",
                     f"{gx:.3f}",
                     f"{gy:.3f}"]
                )

        gif = f"animation_{sid}.gif"
        with imageio.get_writer(
            os.path.join(folder_path, gif),
            mode="I", fps=1
        ) as w:
            for fr in frames:
                w.append_data(fr)

        self.results[sid] = {
            "png": png,
            "csv": csv_name,
            "anim": gif
        }
        self.progress[sid] = 100

    def _read_ascii_grid(self, path):
        header = {}
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(6):
                k, v = f.readline().split()
                header[k.lower()] = float(v)
            arr = np.loadtxt(f)
        return header, arr

    def _extract_patches(self, img, out_dir, sid):
        os.makedirs(out_dir, exist_ok=True)
        half = Config.PATCH_SIZE//2
        coords = []
        nrows, ncols = img.shape
        for r in range(
            0, nrows-Config.PATCH_SIZE+1,
            Config.PATCH_SIZE
        ):
            for c in range(
                0, ncols-Config.PATCH_SIZE+1,
                Config.PATCH_SIZE
            ):
                coords.append((r, c))
        for r in range(
            0,
            nrows-Config.PATCH_SIZE-half+1,
            Config.PATCH_SIZE
        ):
            for c in range(
                0,
                ncols-Config.PATCH_SIZE-half+1,
                Config.PATCH_SIZE
            ):
                coords.append((r+half, c+half))
        total = len(coords)

        def save_rgb(patch, fn):
            rgb = np.stack([patch]*3, -1)
            plt.imsave(fn, rgb, vmin=0, vmax=1)

        for i, (r, c) in enumerate(coords, start=1):
            patch = img[
                r:r+Config.PATCH_SIZE,
                c:c+Config.PATCH_SIZE
            ]
            fn = os.path.join(
                out_dir,
                f"patch_{r}_{c}_{Config.PATCH_SIZE}.png"
            )
            save_rgb(patch, fn)
            self.progress[sid] = math.floor(i * 50 / total)

        self.progress[sid] = 50
        return total

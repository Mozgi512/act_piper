import argparse
import csv
import glob
import os

import cv2
import h5py
import numpy as np


def decode_str(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def build_indep_mask(segments, num_frames):
    indep = np.zeros(num_frames, dtype=bool)
    for seg in segments:
        seg_type = decode_str(seg["type"])
        if seg_type != "independent":
            continue
        start = max(0, int(seg["start"]))
        end = min(num_frames, int(seg["end"]))
        if start < end:
            indep[start:end] = True
    return indep


def detect_red_components(image_bgr, min_area=50):
    b = image_bgr[:, :, 0]
    g = image_bgr[:, :, 1]
    r = image_bgr[:, :, 2]

    mask_bool = (r > 200) & (g < 50) & (b < 50)
    mask = (mask_bool.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label]
        components.append((float(cx), float(cy), area))

    return components, mask


def pick_left_right_red(components, width):
    center_x = width / 2.0
    left = [(cx, cy, area) for (cx, cy, area) in components if cx < center_x]
    right = [(cx, cy, area) for (cx, cy, area) in components if cx >= center_x]

    left_best = None
    if left:
        left_best = max(left, key=lambda t: (t[0], t[2]))

    right_best = None
    if right:
        right_best = min(right, key=lambda t: (t[0], -t[2]))

    return left_best, right_best


def extract_episode(file_path, min_area=50, save_debug_dir=None):
    rows = []
    with h5py.File(file_path, "r") as f:
        if "metadata/left_segments" not in f or "metadata/right_segments" not in f:
            return rows
        if "observations/images/top" not in f:
            return rows

        images = f["observations/images/top"]
        num_frames = images.shape[0]

        left_segs = f["metadata/left_segments"][()]
        right_segs = f["metadata/right_segments"][()]

        left_indep = build_indep_mask(left_segs, num_frames)
        right_indep = build_indep_mask(right_segs, num_frames)

        both_indep = left_indep & right_indep
        indep_starts = np.where(both_indep & np.concatenate(([True], ~both_indep[:-1])))[0]

        episode_name = os.path.splitext(os.path.basename(file_path))[0]

        for indep_idx, t in enumerate(indep_starts.tolist()):
            img_rgb = images[t]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w = img_rgb.shape[:2]

            comps, _ = detect_red_components(img_bgr, min_area=min_area)
            left_best, right_best = pick_left_right_red(comps, w)

            row = {
                "episode": episode_name,
                "frame": int(t),
                "indep_start_index": int(indep_idx),
                "left_red_x": "",
                "left_red_y": "",
                "left_red_area": "",
                "left_red_x_m": "",
                "right_red_x": "",
                "right_red_y": "",
                "right_red_area": "",
                "right_red_x_m": "",
                "red_detect_count": int(len(comps)),
                "image_width": int(w),
                "image_height": int(h),
            }

            if left_best is not None:
                row["left_red_x"] = round(left_best[0], 2)
                row["left_red_y"] = round(left_best[1], 2)
                row["left_red_area"] = int(left_best[2])
                row["left_red_x_m"] = round((float(left_best[0]) - (w / 2.0)) / 224.0, 4)

            if right_best is not None:
                row["right_red_x"] = round(right_best[0], 2)
                row["right_red_y"] = round(right_best[1], 2)
                row["right_red_area"] = int(right_best[2])
                row["right_red_x_m"] = round((float(right_best[0]) - (w / 2.0)) / 224.0, 4)

            rows.append(row)

            if save_debug_dir is not None:
                os.makedirs(save_debug_dir, exist_ok=True)
                dbg = img_bgr.copy()

                box_w = 224
                box_h = 100
                cx = w // 2
                cy = h // 2
                x1 = max(0, cx - box_w // 2)
                y1 = max(0, cy - box_h // 2)
                x2 = min(w - 1, cx + box_w // 2)
                y2 = min(h - 1, cy + box_h // 2)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if left_best is not None:
                    cv2.circle(dbg, (int(left_best[0]), int(left_best[1])), 6, (0, 0, 255), -1)
                if right_best is not None:
                    cv2.circle(dbg, (int(right_best[0]), int(right_best[1])), 6, (0, 0, 255), -1)

                cv2.putText(
                    dbg,
                    f"{episode_name} t={t}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                out_path = os.path.join(save_debug_dir, f"{episode_name}_indep{indep_idx}_t{t}.png")
                cv2.imwrite(out_path, dbg)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract left/right red positions at independent-start frames from HDF5 episodes.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing episode_*.hdf5")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--min_area", type=int, default=50, help="Minimum connected-component area for red detection")
    parser.add_argument("--save_debug_dir", type=str, default=None, help="Optional folder to save annotated debug images")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "episode_*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {args.dataset_dir}")

    all_rows = []
    for file_path in files:
        try:
            rows = extract_episode(file_path, min_area=args.min_area, save_debug_dir=args.save_debug_dir)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[WARN] Failed on {file_path}: {exc}")

    fieldnames = [
        "episode",
        "frame",
        "indep_start_index",
        "left_red_x",
        "left_red_y",
        "left_red_area",
        "left_red_x_m",
        "right_red_x",
        "right_red_y",
        "right_red_area",
        "right_red_x_m",
        "red_detect_count",
        "image_width",
        "image_height",
    ]

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done. episodes={len(files)} rows={len(all_rows)} -> {args.output_csv}")


if __name__ == "__main__":
    main()

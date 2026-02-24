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


def build_coop_mask(segments, num_frames):
    coop = np.zeros(num_frames, dtype=bool)
    for seg in segments:
        seg_type = decode_str(seg["type"])
        if seg_type != "cooperative":
            continue
        start = max(0, int(seg["start"]))
        end = min(num_frames, int(seg["end"]))
        if start < end:
            coop[start:end] = True
    return coop


def detect_color_centroid_bgr(image_bgr, color="green", min_area=30):
    b = image_bgr[:, :, 0]
    g = image_bgr[:, :, 1]
    r = image_bgr[:, :, 2]

    if color == "green":
        # User rule: G > 200 and R/B < 100
        mask_bool = (g > 200) & (r < 50) & (b < 50)
    elif color == "red":
        # Symmetric rule for red: R > 200 and G/B < 100
        mask_bool = (r > 200) & (g < 50) & (b < 50)
    elif color == "blue":
        # Symmetric rule for blue: B > 200 and R/G < 100
        mask_bool = (b > 200) & (r < 50) & (g < 50)
    else:
        raise ValueError(f"Unknown color: {color}")

    mask = (mask_bool.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best = None
    best_x = -1e9
    best_area = -1
    count_valid = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        count_valid += 1
        cx, cy = centroids[label]
        # Choose the right-most component; tie-break by larger area.
        if (cx > best_x) or (cx == best_x and area > best_area):
            best_x = float(cx)
            best_area = area
            best = (float(cx), float(cy), area)

    return best, count_valid, mask


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

        left_coop = build_coop_mask(left_segs, num_frames)
        right_coop = build_coop_mask(right_segs, num_frames)

        both_coop = left_coop & right_coop
        coop_starts = np.where(both_coop & np.concatenate(([True], ~both_coop[:-1])))[0]

        episode_name = os.path.splitext(os.path.basename(file_path))[0]

        for coop_idx, t in enumerate(coop_starts.tolist()):
            img_rgb = images[t]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            g_best, g_count, _ = detect_color_centroid_bgr(img_bgr, color="green", min_area=min_area)
            r_best, r_count, _ = detect_color_centroid_bgr(img_bgr, color="red", min_area=min_area)

            h, w = img_rgb.shape[:2]
            row = {
                "episode": episode_name,
                "frame": int(t),
                "coop_start_index": int(coop_idx),
                "green_x": "",
                "green_y": "",
                "green_area": "",
                "green_detect_count": int(g_count),
                "green_x_m": "",
                "red_x": "",
                "red_y": "",
                "red_area": "",
                "red_detect_count": int(r_count),
                "red_x_m": "",
                "image_width": int(w),
                "image_height": int(h),
            }

            if g_best is not None:
                row["green_x"] = round(g_best[0], 2)
                row["green_y"] = round(g_best[1], 2)
                row["green_area"] = int(g_best[2])
                row["green_x_m"] = round((float(g_best[0]) - (w / 2.0)) / 224.0, 4)

            if r_best is not None:
                row["red_x"] = round(r_best[0], 2)
                row["red_y"] = round(r_best[1], 2)
                row["red_area"] = int(r_best[2])
                row["red_x_m"] = round((float(r_best[0]) - (w / 2.0)) / 224.0, 4)

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
                if g_best is not None:
                    cv2.circle(dbg, (int(g_best[0]), int(g_best[1])), 6, (0, 255, 0), -1)
                if r_best is not None:
                    cv2.circle(dbg, (int(r_best[0]), int(r_best[1])), 6, (0, 0, 255), -1)
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
                out_path = os.path.join(save_debug_dir, f"{episode_name}_coop{coop_idx}_t{t}.png")
                cv2.imwrite(out_path, dbg)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract G/B positions at cooperative-start frames from HDF5 episodes.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing episode_*.hdf5")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--min_area", type=int, default=50, help="Minimum connected-component area for color detection")
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
        "coop_start_index",
        "green_x",
        "green_y",
        "green_area",
        "green_detect_count",
        "green_x_m",
        "red_x",
        "red_y",
        "red_area",
        "red_detect_count",
        "red_x_m",
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

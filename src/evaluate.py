#!/usr/bin/env python3
"""
Skeleton Accuracy Evaluation — using SAM hand masks as pseudo-ground-truth.

Metrics:
  1. Joint-in-Mask (JIM)        — fraction of joints inside the mask
  2. Skeleton–Mask Bbox IoU     — bounding box alignment
  3. Wrist–Centroid Distance    — wrist proximity to mask center
  4. Detection Coverage         — fraction of mask frames with skeleton
  5. Temporal Jitter            — frame-to-frame wrist displacement
  6. Source Breakdown           — all metrics grouped by source_model

Usage:
    conda activate sam3
    python src/evaluate.py data/input.mp4
    python src/evaluate.py data/input.mp4 --seconds 10
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────

MASK_IDX_LH = 0   # Left Hand channel in .npy stack
MASK_IDX_RH = 1   # Right Hand channel

PALM_JOINTS = {0, 1, 5, 9, 13, 17}
FINGERTIP_JOINTS = {4, 8, 12, 16, 20}

DILATE_MARGIN_PX = 8   # pixels to dilate mask for "forgiving" JIM

# ── Utility functions ──────────────────────────────────────────────────

def load_npy_mask(mask_dir, frame_idx):
    """Load [4, H, W] bool mask array from .npy file. Returns None if missing."""
    path = os.path.join(mask_dir, f"frame_{frame_idx:06d}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)  # [4, H, W] bool


def mask_to_bbox(mask_2d):
    """Convert a bool mask to [x1, y1, x2, y2] bbox, or None if empty."""
    if mask_2d is None or not mask_2d.any():
        return None
    ys, xs = np.where(mask_2d)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid(mask_2d):
    """Return (cx, cy) or None."""
    if mask_2d is None or not mask_2d.any():
        return None
    ys, xs = np.where(mask_2d)
    return (float(xs.mean()), float(ys.mean()))


def bbox_iou(a, b):
    """IoU between two [x1, y1, x2, y2] bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def dilate_mask(mask, px):
    """Dilate a bool mask by px pixels."""
    if px <= 0 or not mask.any():
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def bbox_diagonal(bbox):
    """Diagonal length of a bbox."""
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5


# ── Per-frame evaluation ──────────────────────────────────────────────

def evaluate_hand(joints_2d, mask_2d, mask_dilated, prev_wrist):
    """Evaluate a single hand skeleton against its mask.

    Args:
        joints_2d: list of (x, y) tuples, length 21
        mask_2d: bool array [H, W]
        mask_dilated: dilated bool array [H, W]
        prev_wrist: (x, y) from previous frame, or None

    Returns:
        dict of per-hand metrics
    """
    H, W = mask_2d.shape
    n_joints = len(joints_2d)

    # ── Metric 1: Joint-in-Mask ────────────────────────────────────
    def count_in_mask(mask):
        n = 0
        for (x, y) in joints_2d:
            ix, iy = int(x), int(y)
            if 0 <= iy < H and 0 <= ix < W and mask[iy, ix]:
                n += 1
        return n

    jim_raw = count_in_mask(mask_2d) / n_joints if n_joints > 0 else 0.0
    jim_dilated = count_in_mask(mask_dilated) / n_joints if n_joints > 0 else 0.0

    # Palm vs fingertip breakdown (raw mask)
    palm_in = sum(1 for j, (x, y) in enumerate(joints_2d)
                  if j in PALM_JOINTS
                  and 0 <= int(y) < H and 0 <= int(x) < W
                  and mask_2d[int(y), int(x)])
    tips_in = sum(1 for j, (x, y) in enumerate(joints_2d)
                  if j in FINGERTIP_JOINTS
                  and 0 <= int(y) < H and 0 <= int(x) < W
                  and mask_2d[int(y), int(x)])
    jim_palm = palm_in / len(PALM_JOINTS) if len(PALM_JOINTS) > 0 else 0.0
    jim_tips = tips_in / len(FINGERTIP_JOINTS) if len(FINGERTIP_JOINTS) > 0 else 0.0

    # ── Metric 2: Bbox IoU ─────────────────────────────────────────
    xs = [p[0] for p in joints_2d]
    ys = [p[1] for p in joints_2d]
    skel_bbox = [min(xs), min(ys), max(xs), max(ys)]
    mask_bbox = mask_to_bbox(mask_2d)
    iou = bbox_iou(skel_bbox, mask_bbox) if mask_bbox else 0.0

    # ── Metric 3: Wrist–Centroid distance ──────────────────────────
    wrist = joints_2d[0]
    centroid = mask_centroid(mask_2d)
    if centroid and mask_bbox:
        diag = bbox_diagonal(mask_bbox)
        wrist_dist_px = ((wrist[0] - centroid[0]) ** 2 +
                         (wrist[1] - centroid[1]) ** 2) ** 0.5
        wrist_dist_norm = wrist_dist_px / diag if diag > 0 else 0.0
    else:
        wrist_dist_px = 0.0
        wrist_dist_norm = 0.0

    # ── Metric 5: Temporal jitter ──────────────────────────────────
    jitter_px = None
    if prev_wrist is not None:
        jitter_px = ((wrist[0] - prev_wrist[0]) ** 2 +
                     (wrist[1] - prev_wrist[1]) ** 2) ** 0.5

    return {
        "jim_raw": round(jim_raw, 4),
        "jim_dilated": round(jim_dilated, 4),
        "jim_palm": round(jim_palm, 4),
        "jim_tips": round(jim_tips, 4),
        "bbox_iou": round(iou, 4),
        "wrist_dist_px": round(wrist_dist_px, 1),
        "wrist_dist_norm": round(wrist_dist_norm, 4),
        "jitter_px": round(jitter_px, 1) if jitter_px is not None else None,
    }


# ── Aggregation ───────────────────────────────────────────────────────

def aggregate_metrics(per_frame_results):
    """Aggregate per-frame metrics into summary statistics."""
    if not per_frame_results:
        return {}

    keys = ["jim_raw", "jim_dilated", "jim_palm", "jim_tips",
            "bbox_iou", "wrist_dist_px", "wrist_dist_norm"]

    summary = {}
    for k in keys:
        vals = [r[k] for r in per_frame_results if r[k] is not None]
        if vals:
            arr = np.array(vals)
            summary[k] = {
                "mean": round(float(arr.mean()), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(arr.std()), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
            }

    # Jitter separately (has Nones for first frame)
    jitter_vals = [r["jitter_px"] for r in per_frame_results
                   if r["jitter_px"] is not None]
    if jitter_vals:
        arr = np.array(jitter_vals)
        summary["jitter_px"] = {
            "mean": round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "max": round(float(arr.max()), 1),
        }

    summary["n_frames"] = len(per_frame_results)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate skeleton accuracy using SAM hand masks "
                    "as pseudo-ground-truth.")
    p.add_argument("input", help="Input video path (used to derive output names)")
    p.add_argument("--masks", default=None,
                   help="Path to mask directory (default: outputs/<base>_masks/)")
    p.add_argument("--skeleton", default=None,
                   help="Path to skeleton JSON (default: outputs/<base>_skeleton.json)")
    p.add_argument("--seconds", type=float, default=0,
                   help="Evaluate only first N seconds (0 = all)")
    p.add_argument("--dilate", type=int, default=DILATE_MARGIN_PX,
                   help=f"Dilation margin for forgiving JIM (default: {DILATE_MARGIN_PX}px)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: outputs/)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Derive paths
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or "outputs"
    mask_dir = args.masks or os.path.join(out_dir, f"{base}_masks")
    skel_path = args.skeleton or os.path.join(out_dir, f"{base}_skeleton.json")

    eval_json_path = os.path.join(out_dir, f"{base}_eval.json")
    eval_txt_path = os.path.join(out_dir, f"{base}_eval_summary.txt")

    print("=" * 65)
    print("Skeleton Accuracy Evaluation")
    print("  Masks as pseudo-ground-truth")
    print("=" * 65)

    # Validate inputs
    if not os.path.isdir(mask_dir):
        sys.exit(f"[ERROR] Mask directory not found: {mask_dir}\n"
                 f"  Run segment.py first to generate masks.")
    if not os.path.isfile(skel_path):
        sys.exit(f"[ERROR] Skeleton JSON not found: {skel_path}\n"
                 f"  Run skeleton.py first to generate skeletons.")

    # Load skeleton data
    print(f"\n  Masks:    {mask_dir}/")
    print(f"  Skeleton: {skel_path}")

    with open(skel_path) as f:
        skel_data = json.load(f)

    # Get video info for frame limiting
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_frame = total_frames
    if args.seconds > 0:
        max_frame = min(int(args.seconds * fps), total_frames)
        print(f"  Limiting: {args.seconds}s → {max_frame} frames")

    # Build skeleton lookup: frame_idx → {side: hand_data}
    skel_by_frame = {}
    for entry in skel_data:
        fidx = entry["frame"]
        if fidx >= max_frame:
            continue
        hands = {}
        for h in entry["hands"]:
            if len(h.get("landmarks_2d", [])) == 21:
                joints = [(lm["x"], lm["y"]) for lm in h["landmarks_2d"]]
                hands[h["side"]] = {
                    "joints": joints,
                    "source": h.get("source_model", "unknown"),
                }
        if hands:
            skel_by_frame[fidx] = hands

    # Count mask frames
    npy_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
    mask_frame_indices = []
    for f in npy_files:
        fidx = int(f.replace("frame_", "").replace(".npy", ""))
        if fidx < max_frame:
            mask_frame_indices.append(fidx)

    print(f"  Mask frames:     {len(mask_frame_indices)}")
    print(f"  Skeleton frames: {len(skel_by_frame)}")
    print(f"  Dilate margin:   {args.dilate}px")

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\n[1/2] Evaluating per-frame metrics...")
    t0 = time.time()

    # Per-side tracking
    all_results = []     # flat list of per-hand per-frame dicts
    prev_wrist = {"left": None, "right": None}

    # Coverage counters
    mask_count = {"left": 0, "right": 0}
    skel_count = {"left": 0, "right": 0}

    # Source breakdown
    source_results = defaultdict(list)

    for fidx in sorted(mask_frame_indices):
        masks_stack = load_npy_mask(mask_dir, fidx)
        if masks_stack is None:
            continue

        skel_hands = skel_by_frame.get(fidx, {})

        for side, mask_idx in [("left", MASK_IDX_LH), ("right", MASK_IDX_RH)]:
            mask_2d = masks_stack[mask_idx]
            if not mask_2d.any():
                continue

            mask_count[side] += 1

            if side not in skel_hands:
                continue

            skel_count[side] += 1
            hand = skel_hands[side]
            joints = hand["joints"]
            source = hand["source"]

            # Dilate mask for forgiving metric
            mask_dil = dilate_mask(mask_2d, args.dilate)

            metrics = evaluate_hand(joints, mask_2d, mask_dil, prev_wrist[side])
            metrics["frame"] = fidx
            metrics["side"] = side
            metrics["source"] = source

            all_results.append(metrics)
            source_results[source].append(metrics)

            prev_wrist[side] = joints[0]

    elapsed = time.time() - t0
    print(f"  Evaluated {len(all_results)} hand-frame pairs in {elapsed:.1f}s")

    # ── Aggregate ─────────────────────────────────────────────────────
    print(f"\n[2/2] Aggregating results...")

    # Overall
    overall = aggregate_metrics(all_results)

    # Per-side
    left_results = [r for r in all_results if r["side"] == "left"]
    right_results = [r for r in all_results if r["side"] == "right"]
    per_side = {
        "left": aggregate_metrics(left_results),
        "right": aggregate_metrics(right_results),
    }

    # Coverage
    coverage = {}
    for side in ["left", "right"]:
        mc = mask_count[side]
        sc = skel_count[side]
        coverage[side] = {
            "mask_frames": mc,
            "skeleton_frames": sc,
            "coverage": round(sc / mc, 4) if mc > 0 else 0.0,
        }

    # Per-source
    per_source = {}
    for src, results in sorted(source_results.items()):
        per_source[src] = aggregate_metrics(results)

    # ── Format output ─────────────────────────────────────────────────
    summary = {
        "input": args.input,
        "mask_dir": mask_dir,
        "skeleton_json": skel_path,
        "dilate_px": args.dilate,
        "total_hand_frames_evaluated": len(all_results),
        "overall": overall,
        "per_side": per_side,
        "coverage": coverage,
        "per_source": per_source,
    }

    # Save JSON
    with open(eval_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Console + text report ─────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("SKELETON ACCURACY REPORT")
    lines.append(f"  Input:    {args.input}")
    lines.append(f"  Evaluated {len(all_results)} hand-frame pairs")
    lines.append(f"  Dilate:   {args.dilate}px")
    lines.append("=" * 65)

    # Coverage
    lines.append("\n── Detection Coverage ──────────────────────────────────")
    for side in ["left", "right"]:
        c = coverage[side]
        pct = c["coverage"] * 100
        lines.append(f"  {side.capitalize():6s}: {c['skeleton_frames']}/{c['mask_frames']} "
                      f"frames ({pct:.1f}%)")

    # Overall metrics
    lines.append("\n── Overall Metrics ─────────────────────────────────────")
    if overall:
        lines.append(f"  {'Metric':25s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        for key in ["jim_raw", "jim_dilated", "jim_palm", "jim_tips",
                     "bbox_iou", "wrist_dist_norm"]:
            if key in overall:
                s = overall[key]
                lines.append(f"  {key:25s} {s['mean']:8.3f} {s['median']:8.3f} {s['std']:8.3f}")
        if "wrist_dist_px" in overall:
            s = overall["wrist_dist_px"]
            lines.append(f"  {'wrist_dist_px':25s} {s['mean']:8.1f} {s['median']:8.1f} {s['std']:8.1f}")
        if "jitter_px" in overall:
            j = overall["jitter_px"]
            lines.append(f"\n  Temporal Jitter (wrist displacement per frame):")
            lines.append(f"    Mean: {j['mean']:.1f}px  Median: {j['median']:.1f}px  "
                          f"P95: {j['p95']:.1f}px  Max: {j['max']:.1f}px")

    # Per-side
    lines.append("\n── Per-Side Breakdown ──────────────────────────────────")
    for side in ["left", "right"]:
        ps = per_side[side]
        if not ps:
            lines.append(f"  {side.capitalize()}: No data")
            continue
        lines.append(f"  {side.capitalize()} Hand ({ps.get('n_frames', 0)} frames):")
        for key in ["jim_raw", "jim_dilated", "bbox_iou"]:
            if key in ps:
                s = ps[key]
                lines.append(f"    {key:22s}  mean={s['mean']:.3f}  median={s['median']:.3f}")
        if "jitter_px" in ps:
            j = ps["jitter_px"]
            lines.append(f"    jitter_px              mean={j['mean']:.1f}  p95={j['p95']:.1f}")

    # Per-source
    lines.append("\n── Source Model Breakdown ──────────────────────────────")
    lines.append(f"  {'Source':20s} {'N':>5s} {'JIM_raw':>8s} {'JIM_dil':>8s} "
                  f"{'IoU':>8s} {'Jitter':>8s}")
    lines.append(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for src, ps in sorted(per_source.items()):
        n = ps.get("n_frames", 0)
        jr = ps.get("jim_raw", {}).get("mean", 0)
        jd = ps.get("jim_dilated", {}).get("mean", 0)
        io = ps.get("bbox_iou", {}).get("mean", 0)
        jt = ps.get("jitter_px", {}).get("mean", 0) if "jitter_px" in ps else 0
        lines.append(f"  {src:20s} {n:5d} {jr:8.3f} {jd:8.3f} {io:8.3f} {jt:8.1f}")

    # Worst frames
    lines.append("\n── Worst Frames (lowest JIM_raw) ───────────────────────")
    sorted_by_jim = sorted(all_results, key=lambda r: r["jim_raw"])
    for r in sorted_by_jim[:10]:
        lines.append(f"  frame {r['frame']:5d}  {r['side']:5s}  "
                      f"JIM={r['jim_raw']:.3f}  IoU={r['bbox_iou']:.3f}  "
                      f"src={r['source']}")

    lines.append(f"\n{'=' * 65}")
    lines.append(f"  Full report:  {eval_txt_path}")
    lines.append(f"  JSON data:    {eval_json_path}")
    lines.append(f"{'=' * 65}")

    report = "\n".join(lines)
    print(report)

    with open(eval_txt_path, "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()

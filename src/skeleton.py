#!/usr/bin/env python3
"""
Apache 2.0 Hand Skeleton Pipeline (v30/v31)

Fully Apache 2.0 licensed stack — zero MANO, zero HaPTIC.

Architecture:
  - SAM 3 masks  → side discriminator (L/R by mask x-center)
  - MediaPipe    → primary 2D hand landmarks (num_hands=5, lowered confidence)
  - ViTPose+     → fallback wholebody hand keypoints (when MP misses)
  - Mask-pick    → select detection whose wrist falls in side's SAM mask

Phase 1: Core dual-source detection (MP primary + ViTPose+ Huge fallback)

Usage:
    conda activate sam3
    python apache_hand_skeleton.py video.mp4 video_hand_bboxes.json
    python apache_hand_skeleton.py video.mp4 video_hand_bboxes.json -o out.mp4 --no-labels
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from transformers import AutoProcessor, VitPoseForPoseEstimation

# ── Hand skeleton topology (21 joints, MANO/COCO-Wholebody order) ──────
# 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (5, 9), (9, 13), (13, 17),               # Palm cross-bar
]

# 21 landmark names (MediaPipe / MANO joint ordering)
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Fingertip joint indices — get larger dots in rendering
FINGERTIP_INDICES = {4, 8, 12, 16, 20}

# Colors per hand (BGR for OpenCV) — matches track_hands.py
HAND_COLORS = [
    (255, 120, 50),   # blue-ish  (left)
    (50, 180, 255),   # orange-ish (right)
]

# Resolution-adaptive rendering: all sizes scale by max(w,h) / DRAW_SCALE_REF
DRAW_SCALE_REF = 3840

# Mask overlay settings (matching track_hands.py)
MASK_COLORS = {
    "Left Hand":  (255, 120, 50),
    "Right Hand": (50, 180, 255),
    "Left Arm":   (200, 80, 30),
    "Right Arm":  (30, 140, 200),
}
MASK_ALPHA = 0.35

# MediaPipe model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Crop padding factor
DEFAULT_CROP_PAD = 0.35

# Multi-scale wrist crop sizes for ViTPose-assisted retry
WRIST_CROP_SCALES = [300, 450, 600]

# Max gap length (frames) for temporal interpolation
MAX_INTERP_GAP = 15  # ~0.5s at 30fps

# Max wrist drift for interpolation (fraction of image width per frame)
MAX_INTERP_WRIST_DRIFT = 0.05

# Background fill color for mask-applied crops (BGR)
MASK_BG_COLOR = (128, 128, 128)  # neutral gray

# MediaPipe quality thresholds (from reference pipeline)
MP_MIN_HANDEDNESS_SCORE = 0.60       # below this → always reject skeleton
MP_WARN_HANDEDNESS_SCORE = 0.70      # below this → check landmark spread
MP_MIN_LANDMARK_SPREAD = 0.45        # spread < this with low score → reject

# Grip distortion: when skeleton bbox diagonal > this fraction of frame short-dim,
# landmarks are likely scrambled (hand gripping object). Render palm-only.
GRIP_DISTORTION_THRESHOLD = 0.40

# Palm-only connections for grip-distorted rendering
PALM_CONNECTIONS = [(0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
PALM_JOINTS = {0, 1, 2, 5, 9, 13, 17}

# Temporal phantom filtering
TEMPORAL_WINDOW_FRAMES = 2   # check ±2 frames for neighbors
MIN_TEMPORAL_NEIGHBORS = 2   # need ≥2 nearby frames with overlapping bbox
MAX_JUMP_DIAG_FRAC = 0.20    # spatial continuity: 20% of diagonal
SPATIAL_LOOKBACK_FRAMES = 5  # frames of spatial history


# ── Utility functions ──────────────────────────────────────────────────

def download_model_if_missing(model_path):
    """Download the MediaPipe HandLandmarker model if not present."""
    if not os.path.exists(model_path):
        print(f"  Downloading MediaPipe HandLandmarker model → '{model_path}'...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("  Download complete.")
        except Exception as e:
            sys.exit(f"[ERROR] Failed to download model: {e}")


def padded_bbox(bbox, pad_frac, img_w, img_h):
    """Expand a [x1, y1, x2, y2] bbox by `pad_frac` of its size, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_frac)
    py = int(bh * pad_frac)
    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    ]


def point_in_bbox(px, py, bbox):
    """Check if a point (px, py) falls inside a [x1, y1, x2, y2] bbox."""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


# Label indices in the stacked .npy masks [4, H, W]
MASK_IDX_LH = 0  # Left Hand
MASK_IDX_RH = 1  # Right Hand
MASK_IDX_LA = 2  # Left Arm
MASK_IDX_RA = 3  # Right Arm


def load_npy_masks(mask_dir, frame_idx):
    """Load [4, H, W] bool mask array from .npy file.
    Returns numpy array or None."""
    if mask_dir is None:
        return None
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


def apply_mask_to_crop(crop_bgr, mask_crop, bg_color=MASK_BG_COLOR):
    """Apply a binary mask to a crop: fill non-hand pixels with bg_color."""
    result = crop_bgr.copy()
    result[~mask_crop] = bg_color
    return result


def assign_side_by_mask(bboxes, img_width):
    """Assign L/R side using SAM3 mask x-center position (side discriminator).

    In egocentric video (camera on wearer's head):
      - Mask on LEFT side of image  → LEFT hand
      - Mask on RIGHT side of image → RIGHT hand

    Returns: dict mapping bbox_index → 'right' or 'left'
    """
    sides = {}
    if len(bboxes) == 1:
        cx = (bboxes[0]["bbox"][0] + bboxes[0]["bbox"][2]) / 2.0
        sides[0] = "left" if cx < img_width / 2 else "right"
    elif len(bboxes) >= 2:
        # Sort by x-center; in egocentric view, left side = left hand
        centers = [(i, (b["bbox"][0] + b["bbox"][2]) / 2.0) for i, b in enumerate(bboxes)]
        centers.sort(key=lambda x: x[1])
        sides[centers[0][0]] = "left"   # leftmost in image = left hand
        sides[centers[1][0]] = "right"  # rightmost in image = right hand
        # Any extra hands: assign by position
        for cx_pair in centers[2:]:
            sides[cx_pair[0]] = "left" if cx_pair[1] < img_width / 2 else "right"
    return sides


# ── Drawing functions ──────────────────────────────────────────────────

def _draw_scale(frame):
    """Compute resolution-adaptive scale factor."""
    h, w = frame.shape[:2]
    return max(w, h) / DRAW_SCALE_REF


def draw_skeleton(frame, landmarks_px, color, draw_labels=True, grip_distorted=False):
    """Draw a 21-joint hand skeleton on the frame (resolution-adaptive).

    When grip_distorted=True, only the palm subset (wrist + finger MCPs) is drawn
    with a closed polygon, matching the reference pipeline's behavior for hands
    gripping objects where finger landmarks are unreliable.
    """
    scale = _draw_scale(frame)
    line_w = max(2, int(8 * scale))
    dot_r = max(3, int(10 * scale))
    tip_r = max(4, int(14 * scale))
    label_scale = max(0.3, 0.8 * scale)

    pts = landmarks_px
    conns = PALM_CONNECTIONS if grip_distorted else HAND_CONNECTIONS
    visible = PALM_JOINTS if grip_distorted else None
    for (idx1, idx2) in conns:
        if idx1 < len(pts) and idx2 < len(pts):
            cv2.line(frame, pts[idx1], pts[idx2], color, line_w, cv2.LINE_AA)
    for j, pt in enumerate(pts):
        if visible and j not in visible:
            continue
        r = tip_r if j in FINGERTIP_INDICES else dot_r
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, max(1, r - 2), (255, 255, 255), -1, cv2.LINE_AA)
        if draw_labels:
            cv2.putText(
                frame, str(j), (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1,
                cv2.LINE_AA,
            )


def draw_text_with_outline(frame, text, pos, font, scale, color, thickness):
    """Draw text with a dark outline for readability."""
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_hand_info(frame, side, source, color, start_y):
    """Draw compact hand info overlay (resolution-adaptive)."""
    scale = _draw_scale(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.9 * scale)
    font_thick = max(1, int(2 * scale))
    lh = max(18, int(40 * scale))
    x0 = 15
    y = start_y
    title = f"{side.capitalize()} hand — {source}"
    draw_text_with_outline(frame, title, (x0, y), font, font_scale, color, font_thick)
    return y + lh + 5


def _compute_hand_label(det, frame_shape, color):
    """Compute label position and text for collision avoidance.
    Returns (text, color, x, y, tw, th, font_scale, font_thick) or None."""
    lm = det.get("landmarks_px", [])
    if len(lm) < 21:
        return None
    h, w = frame_shape[:2]
    scale = max(w, h) / DRAW_SCALE_REF
    font_scale = max(0.4, 1.0 * scale)
    font_thick = max(1, int(2 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    text = f"{det['side'].capitalize()} — {det['source']}"
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
    pad = max(8, int(6 * scale))
    y = min(ys) - pad
    if y - th < 0:
        y = min(h - 2, max(ys) + th + pad)
    x = max(0, min(min(xs), w - tw))
    return (text, color, x, y, tw, th, font_scale, font_thick)


def draw_hand_labels_with_collision(frame, label_items):
    """Draw per-hand labels with vertical collision avoidance.

    label_items: list of tuples from _compute_hand_label().
    """
    h = frame.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Sort by natural y (top-to-bottom)
    items = sorted([li for li in label_items if li is not None], key=lambda it: it[3])
    placed = []  # (x1, y1, x2, y2) of placed labels
    for text, color, x, y, tw, th, fscale, fthick in items:
        x1, y1, x2, y2 = x, y - th, x + tw, y
        gap = max(8, th)
        bumped = True
        while bumped:
            bumped = False
            for px1, py1, px2, py2 in placed:
                if (x1 < px2 and x2 > px1
                        and y1 < py2 + gap and y2 > py1 - gap):
                    shift = (py2 + gap) - y1
                    y1 += shift
                    y2 += shift
                    y += shift
                    if y2 > h - 2:
                        y = h - 2
                        y1 = y - th
                        y2 = y
                    bumped = True
                    break
        draw_text_with_outline(frame, text, (x, y), font, fscale, color, fthick)
        placed.append((x1, y1, x2, y2))


# ── MediaPipe detection ────────────────────────────────────────────────

def run_mediapipe_on_frame(landmarker, frame_bgr, bboxes, pad_frac, img_w, img_h,
                           sides, frame_masks=None):
    """Run MediaPipe HandLandmarker on mask-derived crops.

    For each SAM3 mask bbox:
      1. Pad and crop the hand region
      2. If mask available, apply it (background → gray) so MP sees only the hand
      3. Run MP on the crop
      4. Map landmarks back to full-frame coordinates
      5. Validate: wrist must fall inside the mask bbox
    """
    detections = {}

    for i, hand_info in enumerate(bboxes):
        raw_bbox = hand_info["bbox"]
        crop_bbox = padded_bbox(raw_bbox, pad_frac, img_w, img_h)
        cx1, cy1, cx2, cy2 = crop_bbox

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # Apply SAM3 mask if available — zero out background
        if frame_masks and i < len(frame_masks) and frame_masks[i] is not None:
            mask_full = frame_masks[i]
            mask_crop = mask_full[cy1:cy2, cx1:cx2]
            crop = apply_mask_to_crop(crop, mask_crop)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            continue

        # Multiple MP detections possible (num_hands=5) —
        # pick the one whose wrist falls closest to the mask center
        best_det = None
        best_dist = float("inf")
        mask_cx = (raw_bbox[0] + raw_bbox[2]) / 2.0
        mask_cy = (raw_bbox[1] + raw_bbox[3]) / 2.0

        for det_j in range(len(result.hand_landmarks)):
            hand_lm = result.hand_landmarks[det_j]
            crop_h, crop_w = crop.shape[:2]

            wrist_x = int(hand_lm[0].x * crop_w) + cx1
            wrist_y = int(hand_lm[0].y * crop_h) + cy1

            if not point_in_bbox(wrist_x, wrist_y, raw_bbox):
                continue

            dist = (wrist_x - mask_cx) ** 2 + (wrist_y - mask_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_j = det_j
                best_det = det_j

        if best_det is None:
            continue

        hand_lm = result.hand_landmarks[best_det]
        crop_h, crop_w = crop.shape[:2]

        landmarks_px = []
        for lm in hand_lm:
            px = int(lm.x * crop_w) + cx1
            py = int(lm.y * crop_h) + cy1
            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))
            landmarks_px.append((px, py))

        # Validate: skeleton must cover a reasonable portion of the mask
        if not check_skeleton_coverage(landmarks_px, raw_bbox):
            continue

        world_lm = None
        if result.hand_world_landmarks and best_det < len(result.hand_world_landmarks):
            world_lm = result.hand_world_landmarks[best_det]

        # Capture MP handedness score for quality filtering
        mp_handedness_score = 1.0
        mp_handedness_raw = "Unknown"
        if result.handedness and best_det < len(result.handedness):
            mp_hand = result.handedness[best_det][0]
            mp_handedness_score = mp_hand.score
            mp_handedness_raw = mp_hand.category_name

        # Quality gate: reject low-confidence MP detections
        if mp_handedness_score < MP_MIN_HANDEDNESS_SCORE:
            continue
        if mp_handedness_score < MP_WARN_HANDEDNESS_SCORE:
            # Check landmark spread — clustered landmarks = likely false positive
            xs = [p[0] for p in landmarks_px]
            ys = [p[1] for p in landmarks_px]
            bw = max(raw_bbox[2] - raw_bbox[0], 1)
            bh = max(raw_bbox[3] - raw_bbox[1], 1)
            spread = max((max(xs) - min(xs)) / bw, (max(ys) - min(ys)) / bh)
            if spread < MP_MIN_LANDMARK_SPREAD:
                continue

        # Grip distortion detection: flag if skeleton bbox diagonal is too large
        grip_distorted = False
        short_dim = min(img_w, img_h)
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        if diag / short_dim > GRIP_DISTORTION_THRESHOLD:
            grip_distorted = True

        detections[i] = {
            "landmarks_px": landmarks_px,
            "world_landmarks": world_lm,
            "handedness_score": round(mp_handedness_score, 3),
            "handedness_raw": mp_handedness_raw,
            "grip_distorted": grip_distorted,
            "side": sides.get(i, "unknown"),
            "source": "mediapipe",
        }

    return detections


# ── ViTPose+ body detection (17 keypoints — arm + wrist localization) ──
# COCO body keypoint indices
COCO_L_SHOULDER = 5
COCO_R_SHOULDER = 6
COCO_L_ELBOW = 7
COCO_R_ELBOW = 8
COCO_L_WRIST = 9
COCO_R_WRIST = 10

# Use COCO body expert (dataset_index=0) — the model has 17 body keypoints
VITPOSE_COCO_DATASET_INDEX = 0

# Minimum skeleton bbox coverage of the mask bbox (reject if smaller)
MIN_SKELETON_COVERAGE = 0.15  # skeleton must cover ≥15% of mask bbox area

# Minimum fraction of skeleton joints that must fall inside the hand mask
MIN_MASK_JOINT_FRACTION = 0.40  # reject if <40% of joints are in mask


def run_vitpose_get_wrists(vitpose_model, vitpose_processor, frame_bgr, bboxes, sides, device):
    """Run ViTPose+ Huge BODY model to get wrist positions.

    The model outputs 17 body keypoints (COCO). We extract:
      - L_Wrist (index 9)
      - R_Wrist (index 10)

    Returns: dict mapping side ('left'/'right') → (x, y, score)
    """
    if len(bboxes) == 0:
        return {}

    h, w = frame_bgr.shape[:2]

    # Union all mask bboxes into one person box (egocentric = one person)
    all_x1 = min(b["bbox"][0] for b in bboxes)
    all_y1 = min(b["bbox"][1] for b in bboxes)
    all_x2 = max(b["bbox"][2] for b in bboxes)
    all_y2 = max(b["bbox"][3] for b in bboxes)

    # Add generous padding for the person box
    pad_w = int((all_x2 - all_x1) * 0.5)
    pad_h = int((all_y2 - all_y1) * 0.5)
    person_x1 = max(0, all_x1 - pad_w)
    person_y1 = max(0, all_y1 - pad_h)
    person_x2 = min(w, all_x2 + pad_w)
    person_y2 = min(h, all_y2 + pad_h)

    # COCO format: [x, y, w, h]
    person_box = np.array([[
        person_x1, person_y1,
        person_x2 - person_x1, person_y2 - person_y1,
    ]], dtype=np.float32)

    # Convert frame to PIL RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Run ViTPose+ with COCO body expert (dataset_index=0)
    inputs = vitpose_processor(
        pil_image, boxes=[person_box], return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        ds_idx = torch.tensor(VITPOSE_COCO_DATASET_INDEX, device=device)
        outputs = vitpose_model(**inputs, dataset_index=ds_idx)

    # Post-process
    pose_results = vitpose_processor.post_process_pose_estimation(
        outputs, boxes=[person_box], threshold=0.1
    )

    if not pose_results or len(pose_results) == 0 or len(pose_results[0]) == 0:
        return {}

    person_pose = pose_results[0][0]
    keypoints = person_pose["keypoints"]  # (17, 2)
    scores = person_pose["scores"]        # (17,)

    wrists = {}
    for side_name, kpt_idx in [("left", COCO_L_WRIST), ("right", COCO_R_WRIST)]:
        if kpt_idx < len(keypoints):
            x = int(keypoints[kpt_idx][0].item())
            y = int(keypoints[kpt_idx][1].item())
            s = float(scores[kpt_idx].item())
            if s > 0.2:
                wrists[side_name] = (x, y, s)

    return wrists




def check_skeleton_coverage(landmarks_px, raw_bbox):
    """Check if skeleton landmarks adequately cover the mask bbox.

    Returns True if the skeleton bounding box covers at least
    MIN_SKELETON_COVERAGE of the mask bbox area.
    """
    if len(landmarks_px) < 21:
        return True  # Don't reject partial detections

    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    skel_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

    mask_area = (raw_bbox[2] - raw_bbox[0]) * (raw_bbox[3] - raw_bbox[1])
    if mask_area == 0:
        return True

    return (skel_area / mask_area) >= MIN_SKELETON_COVERAGE


def validate_skeleton_against_mask(landmarks_px, hand_mask, H, W):
    """Check what fraction of skeleton joints fall inside the hand mask.
    Returns fraction_inside (0.0-1.0). Returns 1.0 if no mask available."""
    if hand_mask is None or len(landmarks_px) < 21:
        return 1.0
    n_inside = 0
    for (x, y) in landmarks_px:
        ix, iy = int(x), int(y)
        if 0 <= iy < H and 0 <= ix < W and hand_mask[iy, ix]:
            n_inside += 1
    return n_inside / len(landmarks_px)


def filter_dets_by_mask(dets, frame_masks, sides, H, W, min_frac=MIN_MASK_JOINT_FRACTION):
    """Remove detections where too few joints fall inside the hand mask.
    Returns (filtered_dets, n_rejected)."""
    if frame_masks is None:
        return dets, 0
    filtered = {}
    n_rejected = 0
    for bi, det in dets.items():
        if len(det.get("landmarks_px", [])) < 21:
            filtered[bi] = det  # keep wrist-only / partial
            continue
        mask_i = frame_masks[bi] if bi < len(frame_masks) else None
        frac = validate_skeleton_against_mask(det["landmarks_px"], mask_i, H, W)
        if frac >= min_frac:
            filtered[bi] = det
        else:
            n_rejected += 1
    return filtered, n_rejected




def _try_mp_on_crop(landmarker, frame_bgr, cx1, cy1, cx2, cy2, img_w, img_h,
                    side, source_tag, use_clahe=False, mask_full=None):
    """Run MediaPipe on a crop region. Returns detection dict or None."""
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    # Apply mask if available
    if mask_full is not None:
        mask_crop = mask_full[cy1:cy2, cx1:cx2]
        crop = apply_mask_to_crop(crop, mask_crop)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Optional CLAHE contrast enhancement (helps with uniform-colored gloves)
    if use_clahe:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        crop_rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None

    hand_lm = result.hand_landmarks[0]
    crop_h, crop_w = crop.shape[:2]

    landmarks_px = []
    for lm in hand_lm:
        px = int(lm.x * crop_w) + cx1
        py = int(lm.y * crop_h) + cy1
        px = max(0, min(px, img_w - 1))
        py = max(0, min(py, img_h - 1))
        landmarks_px.append((px, py))

    world_lm = None
    if result.hand_world_landmarks and len(result.hand_world_landmarks) > 0:
        world_lm = result.hand_world_landmarks[0]

    return {
        "landmarks_px": landmarks_px,
        "world_landmarks": world_lm,
        "side": side,
        "source": source_tag,
    }


def run_vitpose_assisted_mp(landmarker, vitpose_model, vitpose_processor,
                            frame_bgr, bboxes, sides, device, img_w, img_h,
                            frame_masks=None):
    """ViTPose-assisted MediaPipe with multi-scale crops + CLAHE.

    Strategy for each SAM3 mask where MediaPipe missed:
      1. Run ViTPose+ body to get wrist position
      2. Try multiple crop scales (300, 450, 600) around the wrist
      3. For each scale, try both raw and CLAHE-enhanced versions
      4. If still no detection, return wrist-only marker
    """
    wrists = run_vitpose_get_wrists(
        vitpose_model, vitpose_processor, frame_bgr, bboxes, sides, device
    )

    if not wrists:
        return {}

    detections = {}

    for i, hand_info in enumerate(bboxes):
        side = sides.get(i, "right")
        if side not in wrists:
            continue

        wrist_x, wrist_y, wrist_score = wrists[side]

        # Check wrist is near the SAM3 mask bbox
        raw_bbox = hand_info["bbox"]
        slack = 100
        expanded = [raw_bbox[0] - slack, raw_bbox[1] - slack,
                    raw_bbox[2] + slack, raw_bbox[3] + slack]
        if not point_in_bbox(wrist_x, wrist_y, expanded):
            continue

        found = False
        # Try multiple crop scales, each with and without CLAHE
        for crop_size in WRIST_CROP_SCALES:
            half = crop_size // 2
            cx1 = max(0, wrist_x - half)
            cy1 = max(0, wrist_y - half)
            cx2 = min(img_w, wrist_x + half)
            cy2 = min(img_h, wrist_y + half)

            # Try raw crop first
            mask_i = frame_masks[i] if frame_masks and i < len(frame_masks) else None
            det = _try_mp_on_crop(
                landmarker, frame_bgr, cx1, cy1, cx2, cy2,
                img_w, img_h, side, "vitpose+mp", use_clahe=False,
                mask_full=mask_i,
            )
            if det:
                detections[i] = det
                found = True
                break

            # Try with CLAHE enhancement
            det = _try_mp_on_crop(
                landmarker, frame_bgr, cx1, cy1, cx2, cy2,
                img_w, img_h, side, "vitpose+mp_clahe", use_clahe=True,
                mask_full=mask_i,
            )
            if det:
                detections[i] = det
                found = True
                break

        if not found:
            detections[i] = {
                "landmarks_px": [(wrist_x, wrist_y)],
                "side": side,
                "source": "vitpose_wrist",
                "wrist_score": wrist_score,
            }

    return detections


def run_mp_fullframe(landmarker, frame_bgr, bboxes, sides, img_w, img_h):
    """Run MediaPipe on the full frame as a last-resort fallback.

    Match detections to SAM3 bboxes by wrist proximity.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return {}

    detections = {}
    used_dets = set()

    for i, hand_info in enumerate(bboxes):
        raw_bbox = hand_info["bbox"]
        mask_cx = (raw_bbox[0] + raw_bbox[2]) / 2.0
        mask_cy = (raw_bbox[1] + raw_bbox[3]) / 2.0

        best_j = None
        best_dist = float("inf")

        for j, hand_lm in enumerate(result.hand_landmarks):
            if j in used_dets:
                continue
            wrist_x = int(hand_lm[0].x * img_w)
            wrist_y = int(hand_lm[0].y * img_h)
            if not point_in_bbox(wrist_x, wrist_y, raw_bbox):
                continue
            dist = (wrist_x - mask_cx) ** 2 + (wrist_y - mask_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is None:
            continue

        used_dets.add(best_j)
        hand_lm = result.hand_landmarks[best_j]
        landmarks_px = []
        for lm in hand_lm:
            px = max(0, min(int(lm.x * img_w), img_w - 1))
            py = max(0, min(int(lm.y * img_h), img_h - 1))
            landmarks_px.append((px, py))

        world_lm = None
        if result.hand_world_landmarks and best_j < len(result.hand_world_landmarks):
            world_lm = result.hand_world_landmarks[best_j]

        detections[i] = {
            "landmarks_px": landmarks_px,
            "world_landmarks": world_lm,
            "side": sides.get(i, "unknown"),
            "source": "mp_fullframe",
        }

    return detections


def run_mp_guided_by_prev(landmarker, frame_bgr, bboxes, sides,
                          prev_positions, img_w, img_h, frame_masks=None):
    """Use previous frame's wrist position to create a guided crop."""
    if not prev_positions:
        return {}

    detections = {}
    for i, hand_info in enumerate(bboxes):
        side = sides.get(i, "right")
        if side not in prev_positions:
            continue

        prev_wx, prev_wy = prev_positions[side]

        # Try multi-scale crops centered on previous wrist
        for crop_size in [350, 500]:
            half = crop_size // 2
            cx1 = max(0, prev_wx - half)
            cy1 = max(0, prev_wy - half)
            cx2 = min(img_w, prev_wx + half)
            cy2 = min(img_h, prev_wy + half)

            mask_i = frame_masks[i] if frame_masks and i < len(frame_masks) else None
            det = _try_mp_on_crop(
                landmarker, frame_bgr, cx1, cy1, cx2, cy2,
                img_w, img_h, side, "mp_prevguide", use_clahe=False,
                mask_full=mask_i,
            )
            if det:
                detections[i] = det
                break

    return detections


# ── Temporal phantom filtering ─────────────────────────────────────────

def _bbox_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
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


def _bbox_center_dist(a, b):
    """Euclidean distance between bbox centers, normalized by avg bbox size."""
    cx_a = (a[0] + a[2]) / 2
    cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2
    cy_b = (b[1] + b[3]) / 2
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    avg_size = ((a[2] - a[0] + a[3] - a[1]) / 2 +
                (b[2] - b[0] + b[3] - b[1]) / 2) / 2
    return dist / max(avg_size, 1)


def temporal_phantom_filter(all_frame_dets, all_frame_bboxes, width, height):
    """Remove isolated phantom detections that lack temporal neighbors.

    For each detection, requires ≥MIN_TEMPORAL_NEIGHBORS frames within
    ±TEMPORAL_WINDOW_FRAMES that have a spatially overlapping bbox.

    Returns (filtered_dets, n_suppressed).
    """
    n_suppressed = 0
    filtered = {}

    all_fidxs = sorted(all_frame_dets.keys())
    if not all_fidxs:
        return all_frame_dets, 0

    for fidx in all_fidxs:
        dets = all_frame_dets[fidx]
        bboxes = all_frame_bboxes.get(fidx, [])
        kept = {}

        for bi, det in dets.items():
            if bi >= len(bboxes):
                kept[bi] = det
                continue
            bbox = bboxes[bi]["bbox"]

            # Count temporal neighbors
            neighbors = 0
            for offset in range(-TEMPORAL_WINDOW_FRAMES, TEMPORAL_WINDOW_FRAMES + 1):
                if offset == 0:
                    continue
                nf = fidx + offset
                n_bboxes = all_frame_bboxes.get(nf, [])
                for nb_info in n_bboxes:
                    nb = nb_info["bbox"]
                    iou = _bbox_iou(bbox, nb)
                    if iou > 0:
                        neighbors += 1
                        break
                    if _bbox_center_dist(bbox, nb) < 2.0:
                        neighbors += 1
                        break

            if neighbors >= MIN_TEMPORAL_NEIGHBORS:
                kept[bi] = det
            else:
                n_suppressed += 1

        filtered[fidx] = kept

    return filtered, n_suppressed


def spatial_continuity_filter(all_frame_dets, all_frame_bboxes, width, height):
    """Remove detections that jump too far from recent hand positions.

    Returns (filtered_dets, n_suppressed).
    """
    diag = (width ** 2 + height ** 2) ** 0.5
    max_jump = MAX_JUMP_DIAG_FRAC * diag
    n_suppressed = 0

    recent_centers = []  # list of per-frame center lists
    all_fidxs = sorted(all_frame_dets.keys())

    for fidx in all_fidxs:
        dets = all_frame_dets[fidx]
        bboxes = all_frame_bboxes.get(fidx, [])
        all_prev = [c for frame_centers in recent_centers for c in frame_centers]

        if all_prev:
            kept = {}
            for bi, det in dets.items():
                if bi >= len(bboxes):
                    kept[bi] = det
                    continue
                bbox = bboxes[bi]["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                min_dist = min(
                    ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    for px, py in all_prev
                )
                if min_dist <= max_jump:
                    kept[bi] = det
                else:
                    n_suppressed += 1
            all_frame_dets[fidx] = kept
            dets = kept

        # Update recent centers
        frame_centers = []
        for bi in dets:
            if bi < len(bboxes):
                bbox = bboxes[bi]["bbox"]
                frame_centers.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
        recent_centers.append(frame_centers)
        if len(recent_centers) > SPATIAL_LOOKBACK_FRAMES:
            recent_centers.pop(0)

    return all_frame_dets, n_suppressed


# ── Temporal interpolation ─────────────────────────────────────────────

def interpolate_landmarks(lm_a, lm_b, t):
    """Linearly interpolate between two landmark lists. t in [0, 1]."""
    if len(lm_a) != len(lm_b) or len(lm_a) != 21:
        return None
    return [
        (int(lm_a[j][0] * (1 - t) + lm_b[j][0] * t),
         int(lm_a[j][1] * (1 - t) + lm_b[j][1] * t))
        for j in range(21)
    ]


def wrist_drift_ok(lm_a, lm_b, gap_len, img_w):
    """Check if wrist drift between anchor frames is within bounds."""
    if len(lm_a) < 1 or len(lm_b) < 1:
        return True
    dx = abs(lm_a[0][0] - lm_b[0][0])
    dy = abs(lm_a[0][1] - lm_b[0][1])
    dist = (dx ** 2 + dy ** 2) ** 0.5
    max_drift = MAX_INTERP_WRIST_DRIFT * img_w * gap_len
    return dist <= max_drift


# ── Main ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apache 2.0 Hand Skeleton Pipeline (v31) — "
                    "MediaPipe primary + ViTPose+ fallback."
    )
    parser.add_argument("input", help="Path to the original input video.")
    parser.add_argument("bboxes", nargs="?", default=None,
                        help="(Legacy) Path to hand bounding-box JSON. "
                             "Not needed when --masks points to .npy masks.")
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: <input>_apache_skeleton.mp4).",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="Output JSON path for hand joint data (default: <input>_apache_skeleton.json).",
    )
    parser.add_argument(
        "--pad", type=float, default=DEFAULT_CROP_PAD,
        help=f"Crop padding factor for MP (default: {DEFAULT_CROP_PAD}).",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Don't draw joint index labels on the skeleton.",
    )
    parser.add_argument(
        "--show-bbox", action="store_true",
        help="Draw SAM3 bounding boxes on the output for debugging.",
    )
    parser.add_argument(
        "--mp-only", action="store_true",
        help="Disable ViTPose+ fallback, use only MediaPipe.",
    )
    parser.add_argument(
        "--masks", default=None,
        help="Path to mask directory with .npy files from track_hands.py."
             " Each file is [4,H,W] bool: [LH, RH, LA, RA].",
    )
    parser.add_argument(
        "--no-masks", action="store_true",
        help="Don't overlay hand/arm masks on the output video.",
    )
    parser.add_argument(
        "--seconds", type=float, default=5.0,
        help="Process only the first N seconds of the video (default: 5). "
             "Use 0 for the full video.",
    )
    args = parser.parse_args()

    base, _ = os.path.splitext(os.path.basename(args.input))
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    if args.output is None:
        args.output = os.path.join(out_dir, f"{base}_skeleton.mp4")
    if args.json_out is None:
        args.json_out = os.path.join(out_dir, f"{base}_skeleton.json")
    # Auto-detect mask dir (new .npy format)
    if args.masks is None:
        auto_mask_dir = os.path.join(out_dir, f"{base}_masks")
        if os.path.isdir(auto_mask_dir):
            args.masks = auto_mask_dir

    return args


def main():
    args = parse_args()

    print("=" * 65)
    print("Apache 2.0 Hand Skeleton Pipeline (v31)")
    print("  MP primary + ViTPose+ Huge fallback • .npy mask input")
    print("=" * 65)

    # ── Determine input mode: .npy masks or legacy bbox JSON ─────────
    use_npy_masks = False
    bbox_frames = {}
    img_w, img_h = 0, 0

    if args.masks and os.path.isdir(args.masks):
        npy_files = sorted([f for f in os.listdir(args.masks) if f.endswith('.npy')])
        if npy_files:
            use_npy_masks = True
            print(f"\n  Masks:  {len(npy_files)} .npy files from {args.masks}/")
            # Load one to get mask dimensions
            sample = np.load(os.path.join(args.masks, npy_files[0]))
            mask_h, mask_w = sample.shape[1], sample.shape[2]
            print(f"  Mask shape: [{sample.shape[0]}, {mask_h}, {mask_w}]")

    if not use_npy_masks and args.bboxes:
        # Legacy mode: load bbox JSON
        if not os.path.isfile(args.bboxes):
            sys.exit(f"[ERROR] Bbox JSON not found: {args.bboxes}")
        with open(args.bboxes) as f:
            bbox_data = json.load(f)
        bbox_frames = bbox_data["frames"]
        img_w, img_h = bbox_data["width"], bbox_data["height"]
        print(f"\n  Bboxes: {len(bbox_frames)} frames from {args.bboxes}")
    elif not use_npy_masks:
        sys.exit("[ERROR] No mask directory or bbox JSON provided.\n"
                 "  Run: python track_hands.py <video> first, then:\n"
                 "  python apache_hand_skeleton.py <video> --masks <video>_masks/")

    # ── Validate input video ─────────────────────────────────────────
    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] Input video not found: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Cap to --seconds
    if args.seconds > 0:
        max_frames = int(args.seconds * fps)
        if max_frames < total_frames:
            total_frames = max_frames
            print(f"  Limiting to {args.seconds}s → {total_frames} frames")

    if use_npy_masks:
        img_w, img_h = width, height
        print(f"  Video:  {args.input}  ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")
    else:
        assert width == img_w and height == img_h, (
            f"Video dimensions ({width}x{height}) don't match bbox JSON ({img_w}x{img_h})"
        )

    # ── Setup MediaPipe ──────────────────────────────────────────────
    print(f"\n[1/4] Setting up MediaPipe HandLandmarker...")
    download_model_if_missing(MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    mp_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=5,  # detect up to 5, then mask-pick
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(mp_options)
    print("  MediaPipe ready (confidence=0.15).")

    # ── Setup ViTPose+ ───────────────────────────────────────────────
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vitpose_model = None
    vitpose_processor = None

    if not args.mp_only:
        print(f"\n[2/4] Loading ViTPose+ Huge wholebody (fallback)...")
        t0 = time.time()
        vitpose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
        vitpose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-huge",
            device_map=device,
        )
        vitpose_model.eval()
        print(f"  ViTPose+ ready on {device} in {time.time() - t0:.1f}s")
    else:
        print(f"\n[2/4] ViTPose+ disabled (--mp-only mode)")

    # ── Process video — Pass 1: Detect ────────────────────────────────
    print(f"\n[3/5] Processing {total_frames} frames (detection pass)...")
    t0 = time.time()

    cap = cv2.VideoCapture(args.input)

    # Per-frame detection results (indexed by frame number)
    all_frame_dets = {}
    all_frame_bboxes = {}
    all_frame_sides = {}
    prev_wrist_positions = {}  # side → (x, y) from previous frame
    stats = {
        "mediapipe": 0, "mp_prevguide": 0, "vitpose+mp": 0,
        "vitpose+mp_clahe": 0, "mp_fullframe": 0, "vitpose_wrist": 0,
        "interp": 0, "frames_with_skeleton": 0,
        "mask_rejected": 0,
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        fidx_str = str(frame_idx)

        # ── Determine bboxes, sides, and masks for this frame ──
        has_data = False
        bboxes = []
        sides = {}
        frame_masks = None

        if use_npy_masks:
            # Load stacked [4, H, W] masks: [LH, RH, LA, RA]
            npy_masks = load_npy_masks(args.masks, frame_idx)
            if npy_masks is not None:
                # Resize masks to video dims if needed
                if npy_masks.shape[1:] != (height, width):
                    resized = []
                    for ch in range(npy_masks.shape[0]):
                        r = cv2.resize(npy_masks[ch].astype(np.uint8),
                                       (width, height),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                        resized.append(r)
                    npy_masks = np.stack(resized)

                # Derive hand bboxes from LH/RH mask channels
                lh_mask = npy_masks[MASK_IDX_LH]
                rh_mask = npy_masks[MASK_IDX_RH]
                lh_bbox = mask_to_bbox(lh_mask)
                rh_bbox = mask_to_bbox(rh_mask)

                frame_masks = []
                for side_name, hand_mask, hand_bbox in [
                    ("left", lh_mask, lh_bbox),
                    ("right", rh_mask, rh_bbox),
                ]:
                    if hand_bbox is not None:
                        idx = len(bboxes)
                        bboxes.append({"bbox": hand_bbox, "obj_id": idx})
                        sides[idx] = side_name
                        frame_masks.append(hand_mask)


                has_data = len(bboxes) > 0

        elif fidx_str in bbox_frames:
            # Legacy mode: use bbox JSON
            bboxes = bbox_frames[fidx_str]
            sides = assign_side_by_mask(bboxes, width)
            has_data = True

            # Load legacy masks if available
            if args.masks:
                frame_masks = []
                for bi, binfo in enumerate(bboxes):
                    obj_id = binfo.get("obj_id", bi)
                    m = load_npy_masks(args.masks, frame_idx)
                    if m is not None and bi < m.shape[0]:
                        frame_masks.append(m[bi])
                    else:
                        frame_masks.append(None)


        if has_data:
            all_frame_bboxes[frame_idx] = bboxes
            all_frame_sides[frame_idx] = sides

            # Stage 1: MediaPipe primary (with mask-applied crops)
            mp_dets = run_mediapipe_on_frame(
                landmarker, frame, bboxes, args.pad, width, height, sides,
                frame_masks=frame_masks,
            )
            mp_dets, n_rej = filter_dets_by_mask(mp_dets, frame_masks, sides, height, width)
            stats["mask_rejected"] += n_rej

            # Stage 2: Previous-frame guided crops
            missed_indices = [i for i in range(len(bboxes)) if i not in mp_dets]
            prev_dets = {}
            if missed_indices and prev_wrist_positions:
                prev_dets = run_mp_guided_by_prev(
                    landmarker, frame, bboxes, sides,
                    prev_wrist_positions, width, height,
                    frame_masks=frame_masks,
                )
                prev_dets = {k: v for k, v in prev_dets.items() if k in missed_indices}
                prev_dets, n_rej = filter_dets_by_mask(prev_dets, frame_masks, sides, height, width)
                stats["mask_rejected"] += n_rej

            # Stage 3: ViTPose-assisted fallback
            still_missed = [i for i in missed_indices if i not in prev_dets]
            vit_dets = {}
            if still_missed and vitpose_model is not None:
                vit_dets = run_vitpose_assisted_mp(
                    landmarker, vitpose_model, vitpose_processor,
                    frame, bboxes, sides, device, width, height,
                    frame_masks=frame_masks,
                )
                vit_dets = {k: v for k, v in vit_dets.items() if k in still_missed}
                vit_dets, n_rej = filter_dets_by_mask(vit_dets, frame_masks, sides, height, width)
                stats["mask_rejected"] += n_rej

            # Stage 4: Full-frame MP fallback
            final_missed = [i for i in still_missed if i not in vit_dets]
            ff_dets = {}
            if final_missed:
                ff_dets = run_mp_fullframe(
                    landmarker, frame, bboxes, sides, width, height,
                )
                ff_dets = {k: v for k, v in ff_dets.items() if k in final_missed}
                ff_dets, n_rej = filter_dets_by_mask(ff_dets, frame_masks, sides, height, width)
                stats["mask_rejected"] += n_rej

            # Merge all
            all_dets = {**mp_dets}
            for extra in [prev_dets, vit_dets, ff_dets]:
                for k, v in extra.items():
                    if k not in all_dets:
                        all_dets[k] = v

            all_frame_dets[frame_idx] = all_dets


            # Update previous wrist positions for next frame
            for bi, det in all_dets.items():
                if len(det["landmarks_px"]) >= 21:
                    prev_wrist_positions[det["side"]] = det["landmarks_px"][0]
                elif len(det["landmarks_px"]) == 1:
                    prev_wrist_positions[det["side"]] = det["landmarks_px"][0]

            # Count stats
            for det in all_dets.values():
                src = det["source"]
                if src in stats:
                    stats[src] += 1

        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            rate = frame_idx / max(elapsed, 0.01)
            skel_count = sum(1 for fd in all_frame_dets.values()
                            for d in fd.values() if len(d["landmarks_px"]) >= 21)
            sys.stdout.write(
                f"\r  {frame_idx}/{total_frames} ({pct:.1f}%) — "
                f"{rate:.1f} FPS — skeletons:{skel_count}"
            )
            sys.stdout.flush()

    cap.release()
    detect_elapsed = time.time() - t0
    print(f"\n  Detection pass done in {detect_elapsed:.1f}s")

    # ── Pass 1.5a: Temporal phantom filtering ─────────────────────────
    print(f"\n[4/6] Temporal phantom filtering (window=±{TEMPORAL_WINDOW_FRAMES})...")
    all_frame_dets, n_phantom = temporal_phantom_filter(
        all_frame_dets, all_frame_bboxes, width, height,
    )
    all_frame_dets, n_spatial = spatial_continuity_filter(
        all_frame_dets, all_frame_bboxes, width, height,
    )
    stats["phantom_suppressed"] = n_phantom
    stats["spatial_suppressed"] = n_spatial
    print(f"  Suppressed {n_phantom} temporal phantoms, {n_spatial} spatial outliers")

    # ── Pass 1.5b: Temporal interpolation ─────────────────────────────
    print(f"\n[5/6] Temporal interpolation (max gap={MAX_INTERP_GAP}, "
          f"max drift={MAX_INTERP_WRIST_DRIFT*100:.0f}%/frame)...")

    # Build per-side timeline: frame → landmarks_px (21 joints only)
    side_timelines = {"left": {}, "right": {}}
    for fidx, dets in all_frame_dets.items():
        for bi, det in dets.items():
            if len(det["landmarks_px"]) == 21:
                side_timelines[det["side"]][fidx] = det["landmarks_px"]

    interp_count = 0
    for side_name in ["left", "right"]:
        timeline = side_timelines[side_name]
        if len(timeline) < 2:
            continue

        sorted_frames = sorted(timeline.keys())
        for k in range(len(sorted_frames) - 1):
            fa = sorted_frames[k]
            fb = sorted_frames[k + 1]
            gap = fb - fa - 1
            if gap <= 0 or gap > MAX_INTERP_GAP:
                continue

            lm_a = timeline[fa]
            lm_b = timeline[fb]

            # Wrist-drift safety check
            if not wrist_drift_ok(lm_a, lm_b, gap, width):
                continue

            for interp_f in range(fa + 1, fb):
                t = (interp_f - fa) / (fb - fa)
                interp_lm = interpolate_landmarks(lm_a, lm_b, t)
                if interp_lm is None:
                    continue

                if interp_f not in all_frame_sides:
                    continue
                sides_f = all_frame_sides[interp_f]
                target_bi = None
                for bi, s in sides_f.items():
                    if s == side_name:
                        target_bi = bi
                        break
                if target_bi is None:
                    continue

                if interp_f not in all_frame_dets:
                    all_frame_dets[interp_f] = {}
                if target_bi not in all_frame_dets[interp_f]:
                    all_frame_dets[interp_f][target_bi] = {
                        "landmarks_px": interp_lm,
                        "side": side_name,
                        "source": "interp",
                    }
                    interp_count += 1
                elif len(all_frame_dets[interp_f][target_bi]["landmarks_px"]) == 1:
                    # Replace wrist-only with interpolated skeleton
                    all_frame_dets[interp_f][target_bi] = {
                        "landmarks_px": interp_lm,
                        "side": side_name,
                        "source": "interp",
                    }
                    interp_count += 1

    stats["interp"] = interp_count
    print(f"  Interpolated {interp_count} hand detections")

    # ── Pass 2: Render video ──────────────────────────────────────────
    print(f"\n[6/6] Rendering output video...")
    t0 = time.time()

    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    tracking_data = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        frame_data = {
            "frame": frame_idx,
            "timestamp_ms": int((frame_idx / fps) * 1000),
            "hands": [],
        }

        # Draw mask overlays (behind everything else)
        if not args.no_masks and use_npy_masks:
            npy_masks = load_npy_masks(args.masks, frame_idx)
            if npy_masks is not None:
                # Resize masks to video dims if needed
                if npy_masks.shape[1:] != (height, width):
                    resized = []
                    for ch in range(npy_masks.shape[0]):
                        r = cv2.resize(npy_masks[ch].astype(np.uint8),
                                       (width, height),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                        resized.append(r)
                    npy_masks = np.stack(resized)

                overlay = frame.copy()
                mask_labels = ["Left Hand", "Right Hand", "Left Arm", "Right Arm"]
                for ch_idx, label in enumerate(mask_labels):
                    if ch_idx < npy_masks.shape[0]:
                        m = npy_masks[ch_idx]
                        if m.any():
                            overlay[m] = MASK_COLORS[label]
                frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

        # Draw bboxes if requested
        if args.show_bbox and frame_idx in all_frame_bboxes:
            bboxes = all_frame_bboxes[frame_idx]
            sides = all_frame_sides.get(frame_idx, {})
            for bi, binfo in enumerate(bboxes):
                bx1, by1, bx2, by2 = binfo["bbox"]
                color = HAND_COLORS[bi % len(HAND_COLORS)]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
                side = sides.get(bi, "?")
                cv2.putText(frame, side, (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if frame_idx in all_frame_dets:
            all_dets = all_frame_dets[frame_idx]
            bboxes = all_frame_bboxes.get(frame_idx, [])
            sides = all_frame_sides.get(frame_idx, {})
            any_drawn = False

            # Draw hand skeletons + collect labels for collision avoidance
            label_items = []
            for bi in sorted(all_dets.keys()):
                det = all_dets[bi]
                color = HAND_COLORS[bi % len(HAND_COLORS)]

                if len(det["landmarks_px"]) >= 21:
                    draw_skeleton(frame, det["landmarks_px"], color,
                                  draw_labels=not args.no_labels,
                                  grip_distorted=det.get("grip_distorted", False))
                elif len(det["landmarks_px"]) == 1:
                    scale = _draw_scale(frame)
                    wr = max(8, int(20 * scale))
                    wx, wy = det["landmarks_px"][0]
                    cv2.circle(frame, (wx, wy), wr, color, max(2, int(5 * scale)), cv2.LINE_AA)
                    cv2.circle(frame, (wx, wy), max(2, int(6 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
                any_drawn = True

                # Compute label for collision-avoiding placement
                label_items.append(_compute_hand_label(det, frame.shape, color))

                # JSON export — richer format with {id, name, x, y, z}
                lm_list = []
                for j, p in enumerate(det["landmarks_px"]):
                    entry = {
                        "id": j,
                        "name": LANDMARK_NAMES[j] if j < len(LANDMARK_NAMES) else f"POINT_{j}",
                        "x": p[0],
                        "y": p[1],
                    }
                    # Add z from world landmarks if available
                    if det.get("world_landmarks") and j < len(det["world_landmarks"]):
                        wl = det["world_landmarks"]
                        # MediaPipe world landmarks are objects with .z attr
                        if hasattr(wl[j], 'z'):
                            entry["z"] = round(wl[j].z, 6)
                    lm_list.append(entry)

                hand_entry = {
                    "obj_id": bboxes[bi].get("obj_id", bi) if bi < len(bboxes) else bi,
                    "side": det["side"],
                    "source_model": det["source"],
                    "handedness_score": det.get("handedness_score", None),
                    "handedness_raw": det.get("handedness_raw", None),
                    "grip_distorted": det.get("grip_distorted", False),
                    "landmarks_2d": lm_list,
                }
                if det["source"] in ("mediapipe", "vitpose+mp", "vitpose+mp_clahe",
                                     "mp_fullframe", "mp_prevguide") and det.get("world_landmarks"):
                    wl = det["world_landmarks"]
                    hand_entry["joints_3d_meters"] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in wl
                    ]
                if det["source"] == "vitpose_wrist":
                    hand_entry["wrist_score"] = det.get("wrist_score", 0.0)


                frame_data["hands"].append(hand_entry)

            # Draw labels with collision avoidance (after all skeletons)
            draw_hand_labels_with_collision(frame, label_items)

            if any_drawn:
                stats["frames_with_skeleton"] += 1

        if frame_data["hands"]:
            tracking_data.append(frame_data)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    render_elapsed = time.time() - t0
    total_real = sum(v for k, v in stats.items()
                     if k not in ("frames_with_skeleton", "mask_rejected"))
    skel_frames = stats["frames_with_skeleton"]
    print(f"  Rendered in {render_elapsed:.1f}s")
    print(f"\n  === Results ===")
    print(f"  Frames with skeleton: {skel_frames}/{frame_idx} ({skel_frames/max(frame_idx,1)*100:.1f}%)")
    print(f"  Detection sources:")
    for src in ["mediapipe", "mp_prevguide", "vitpose+mp", "vitpose+mp_clahe",
                "mp_fullframe", "vitpose_wrist", "interp"]:
        if stats[src] > 0:
            print(f"    {src}: {stats[src]}")
    print(f"  Total detections: {total_real}")
    if stats["mask_rejected"] > 0:
        print(f"  Mask-rejected: {stats['mask_rejected']} "
              f"(skeletons with <{int(MIN_MASK_JOINT_FRACTION*100)}% joints in mask)")
    if stats.get("phantom_suppressed", 0) > 0:
        print(f"  Phantom-suppressed: {stats['phantom_suppressed']}")
    if stats.get("spatial_suppressed", 0) > 0:
        print(f"  Spatial-suppressed: {stats['spatial_suppressed']}")

    # ── Save tracking data ───────────────────────────────────────────
    print(f"\nSaving joint data → {args.json_out}")
    with open(args.json_out, "w") as f:
        json.dump(tracking_data, f, indent=2)

    landmarker.close()

    print(f"\n{'=' * 65}")
    print(f"Output video : {args.output}")
    print(f"Joint data   : {args.json_out}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

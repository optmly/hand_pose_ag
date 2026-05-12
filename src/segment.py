#!/usr/bin/env python3
"""
v0.1 — Hand & Body Segmentation for egocentric video.

Pipeline:
  1. SAM3 image mode (text prompt) → detect hands on prompt frame
  2. SAM2 video mode (mask prompt)  → propagate hand masks through video
  3. SAM2 video mode (point prompt) → segment ego person's body
  4. Assign left/right with temporal consistency
  5. Output video 1: hand masks with L/R labels
  6. Output video 2: body mask with dilated hand masks subtracted
  7. Export bboxes to JSON and masks to .npy stacks

Usage:
    conda activate sam3
    python src/segment.py data/input.mp4
    python src/segment.py data/input.mp4 --seconds 10
    python src/segment.py data/input.mp4 -o outputs/
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DEVWS_DIR = os.path.dirname(_PROJECT_DIR)           # /home/jingjin/devws

# SAM3 lives in the hand_pose_4090 workspace (editable install)
BPE_PATH = os.path.join(_DEVWS_DIR, "hand_pose_4090", "sam3", "sam3",
                        "assets", "bpe_simple_vocab_16e6.txt.gz")

# SAM2 checkpoint
SAM2_CHECKPOINT = os.path.join(_DEVWS_DIR, "sam2_repo", "checkpoints",
                               "sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_HAND_PROMPT = "egocentric first person's hands or gloves"
HAND_DILATE_PX = 30          # dilation radius when subtracting hands from body
MASK_ALPHA = 0.45

COLORS = {
    "Left Hand":  (255, 120, 50),   # orange
    "Right Hand": (50, 180, 255),   # cyan
    "Body":       (80, 200, 80),    # green
}


# ── Utility functions ──────────────────────────────────────────────────

def mask_centroid_x(mask_bool):
    """Return x-coordinate of mask centroid, or None."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return float(xs.mean())


def mask_to_bbox(mask_2d):
    """Convert a bool mask to [x1, y1, x2, y2] bbox, or None if empty."""
    if mask_2d is None or not mask_2d.any():
        return None
    ys, xs = np.where(mask_2d)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid(mask_bool):
    """Return (x, y) centroid or None."""
    if mask_bool is None or not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return (float(xs.mean()), float(ys.mean()))


def dilate_mask(mask, px):
    """Dilate a bool mask by px pixels."""
    if px <= 0 or not mask.any():
        return mask.copy()
    if px > 30:
        dist = cv2.distanceTransform(
            (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return dist <= px
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def extract_frames_jpg(video_path, frames_dir, max_frames=0):
    """Extract video frames as JPEG files (required by SAM2)."""
    if os.path.isdir(frames_dir) and glob.glob(os.path.join(frames_dir, "*.jpg")):
        n = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        print(f"       Reusing {n} existing frames in {frames_dir}")
        return n
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 0 < max_frames <= idx:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"       Extracted {idx} frames → {frames_dir}")
    return idx


# ── SAM3 image-mode hand detection ────────────────────────────────────

def detect_hands_sam3_image(video_path, prompt_frame_idx, prompt_text,
                            confidence_threshold=0.3, max_hands=2):
    """Use SAM3 in IMAGE mode to detect hands on a single frame.

    Returns list of boolean masks (numpy H×W), sorted by centroid-x
    (leftmost first). At most max_hands masks returned.
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image

    # Read the prompt frame from video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, prompt_frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {prompt_frame_idx}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Build SAM3 image model
    t0 = time.time()
    model = build_sam3_image_model(bpe_path=BPE_PATH)
    print(f"       SAM3 image model loaded in {time.time() - t0:.1f}s")

    # Run detection (SAM3 image model requires bfloat16 autocast per official examples)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(state=state, prompt=prompt_text)

    n_detected = len(state["scores"])
    print(f"       Detected {n_detected} object(s) with text: \"{prompt_text}\"")

    if n_detected == 0:
        del model, processor
        torch.cuda.empty_cache()
        return []

    # Extract masks as numpy bool arrays
    masks_tensor = state["masks"]    # shape: (N, 1, H, W), bool
    scores = state["scores"]         # shape: (N,)

    # Convert to numpy (cast from bf16 to f32 first)
    masks_np = masks_tensor.squeeze(1).cpu().numpy().astype(bool)  # (N, H, W)
    scores_np = scores.float().cpu().numpy()

    # Sort by score descending, keep top max_hands
    order = np.argsort(-scores_np)
    masks_np = masks_np[order]
    scores_np = scores_np[order]

    if len(masks_np) > max_hands:
        # Additionally prefer masks in the bottom half (ego hand heuristic)
        def bottom_score(m):
            total = m.sum()
            if total == 0:
                return 0
            mid_y = m.shape[0] // 2
            return (m[mid_y:].sum() / total) * total

        scored = [(bottom_score(masks_np[i]) * scores_np[i], i)
                  for i in range(len(masks_np))]
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_idxs = [s[1] for s in scored[:max_hands]]
        masks_np = masks_np[keep_idxs]

    # Sort remaining by centroid-x (leftmost first)
    cxs = [mask_centroid_x(m) or float('inf') for m in masks_np]
    cx_order = np.argsort(cxs)
    masks_np = masks_np[cx_order]

    for i, m in enumerate(masks_np):
        cx = mask_centroid_x(m)
        print(f"       Hand {i}: {m.sum()} px, cx={cx:.0f}")

    del model, processor
    torch.cuda.empty_cache()

    return list(masks_np)


# ── SAM2 video propagation ────────────────────────────────────────────

def propagate_masks_sam2(frames_dir, prompt_frame_idx, initial_masks,
                         obj_id_start=1):
    """Use SAM2 to propagate initial masks through video frames.

    Args:
        frames_dir: Directory with JPEG frames for SAM2
        prompt_frame_idx: Frame index where initial masks are from
        initial_masks: List of boolean numpy masks (H×W)
        obj_id_start: Starting object ID

    Returns:
        List of {fidx: mask2d} dicts, one per input mask.
    """
    from sam2.build_sam import build_sam2_video_predictor

    t0 = time.time()
    sam2 = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    print(f"       SAM2 loaded in {time.time() - t0:.1f}s")

    results = [{} for _ in initial_masks]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2.init_state(video_path=frames_dir)

        # Add each mask as a separate object
        for i, mask in enumerate(initial_masks):
            obj_id = obj_id_start + i
            # SAM2 expects mask as (1, H, W) or (H, W) tensor
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).cuda()
            sam2.add_new_mask(
                state, frame_idx=prompt_frame_idx, obj_id=obj_id,
                mask=mask_tensor,
            )
            print(f"       Added hand mask {i} as obj_id={obj_id}")

        # Propagate
        t0 = time.time()
        n_frames = 0
        for out_fidx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(state):
            masks_bool = (out_mask_logits > 0.0).squeeze(1).cpu().numpy()
            for j, oid in enumerate(out_obj_ids):
                idx = int(oid) - obj_id_start
                if 0 <= idx < len(results):
                    m = masks_bool[j]
                    if m.any():
                        results[idx][out_fidx] = m
            n_frames += 1
        print(f"       Propagated {n_frames} frames in {time.time() - t0:.1f}s")

    del sam2, state
    torch.cuda.empty_cache()

    return results


def segment_body_sam2(frames_dir, prompt_frame_idx, W, H):
    """Segment ego person's body using SAM2 with point prompt at lower-center."""
    from sam2.build_sam import build_sam2_video_predictor

    t0 = time.time()
    sam2 = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    print(f"       SAM2 loaded in {time.time() - t0:.1f}s")

    body_point = np.array([[W / 2, H * 0.85]], dtype=np.float32)
    body_label = np.array([1], dtype=np.int32)

    body_masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2.init_state(video_path=frames_dir)
        sam2.add_new_points_or_box(
            state, frame_idx=prompt_frame_idx, obj_id=1,
            points=body_point, labels=body_label,
        )
        t0 = time.time()
        for out_fidx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(state):
            m = (out_mask_logits > 0.0).squeeze(1).cpu().numpy()[0]
            if m.any():
                body_masks[out_fidx] = m
        print(f"       SAM2 body: {len(body_masks)} frames in {time.time() - t0:.1f}s")

    del sam2, state
    torch.cuda.empty_cache()

    return body_masks


# ── Left/right assignment with temporal consistency ───────────────────

def assign_left_right(hand_mask_dicts, W):
    """Assign left/right labels to two hand mask dicts.

    In egocentric video, the camera mirrors L/R:
    - Object on the LEFT side of the image → RIGHT hand
    - Object on the RIGHT side of the image → LEFT hand

    Temporal consistency: verify per-frame that L/R don't swap.
    """
    if len(hand_mask_dicts) < 2:
        if len(hand_mask_dicts) == 1:
            all_fidxs = sorted(hand_mask_dicts[0].keys())
            if all_fidxs:
                mid_fidx = all_fidxs[len(all_fidxs) // 2]
                cx = mask_centroid_x(hand_mask_dicts[0][mid_fidx])
                if cx is not None and cx < W / 2:
                    return hand_mask_dicts[0], {}   # right hand (left in image)
                else:
                    return {}, hand_mask_dicts[0]   # left hand (right in image)
        return {}, {}

    masks_a, masks_b = hand_mask_dicts[0], hand_mask_dicts[1]

    def avg_cx(masks_dict):
        cxs = [mask_centroid_x(m) for m in masks_dict.values()
               if mask_centroid_x(m) is not None]
        return np.mean(cxs) if cxs else W / 2

    avg_a, avg_b = avg_cx(masks_a), avg_cx(masks_b)

    # In ego view: left-in-image = right hand
    if avg_a < avg_b:
        right_hand_masks, left_hand_masks = masks_a, masks_b
    else:
        right_hand_masks, left_hand_masks = masks_b, masks_a

    # Temporal consistency: fix per-frame swaps
    all_fidxs = sorted(set(left_hand_masks.keys()) | set(right_hand_masks.keys()))
    n_swaps = 0
    for fidx in all_fidxs:
        lm = left_hand_masks.get(fidx)
        rm = right_hand_masks.get(fidx)
        if lm is None or rm is None:
            continue
        lcx = mask_centroid_x(lm)
        rcx = mask_centroid_x(rm)
        if lcx is not None and rcx is not None and lcx < rcx:
            left_hand_masks[fidx], right_hand_masks[fidx] = rm, lm
            n_swaps += 1

    if n_swaps > 0:
        print(f"       Fixed {n_swaps} L/R swap(s) for temporal consistency")

    return left_hand_masks, right_hand_masks


# ── Video rendering ───────────────────────────────────────────────────

def render_hand_video(video_path, left_masks, right_masks, output_path, fps, W, H):
    """Render video with colored hand masks and L/R labels."""
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for label, masks in [("Left Hand", left_masks), ("Right Hand", right_masks)]:
            m = masks.get(fidx)
            if m is None:
                continue
            mask = m.astype(bool)
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            color = COLORS[label]
            overlay[mask] = color
            mu8 = mask.astype(np.uint8)
            cs, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cs, -1, color, 2)
            if cs:
                M = cv2.moments(cs[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame, label, (cx - 50, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                                cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
        out.write(frame)
        fidx += 1

    cap.release()
    out.release()
    print(f"       Hand video → {output_path}")


def render_body_video(video_path, body_masks, left_masks, right_masks,
                      output_path, fps, W, H, dilate_px=HAND_DILATE_PX):
    """Render video with body mask, dilated hand masks subtracted."""
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bm = body_masks.get(fidx)
        if bm is not None:
            body = bm.astype(bool)
            if body.shape[:2] != (H, W):
                body = cv2.resize(body.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            for hand_masks in [left_masks, right_masks]:
                hm = hand_masks.get(fidx)
                if hm is not None:
                    hand = hm.astype(bool)
                    if hand.shape[:2] != (H, W):
                        hand = cv2.resize(hand.astype(np.uint8), (W, H),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                    body = body & ~dilate_mask(hand, dilate_px)
            overlay = frame.copy()
            overlay[body] = COLORS["Body"]
            frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            mu8 = body.astype(np.uint8)
            cs, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cs, -1, COLORS["Body"], 2)
        out.write(frame)
        fidx += 1

    cap.release()
    out.release()
    print(f"       Body video → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Egocentric hand & body segmentation (SAM3 image + SAM2 video)")
    p.add_argument("input", help="Input video path")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: outputs/)")
    p.add_argument("-p", "--prompt", default=DEFAULT_HAND_PROMPT,
                   help="SAM3 text prompt for hands")
    p.add_argument("--thresh", type=float, default=0.3,
                   help="SAM3 confidence threshold")
    p.add_argument("--prompt-frame", type=int, default=0,
                   help="Frame index for initial detection")
    p.add_argument("--seconds", type=float, default=0,
                   help="Seconds to process (0 = full video)")
    p.add_argument("--hand-dilate", type=int, default=HAND_DILATE_PX,
                   help="Dilation radius for hand subtraction from body")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("=" * 60)
    print("v0.1 — Egocentric Hand & Body Segmentation")
    print("         (SAM3 image + SAM2 video)")
    print("=" * 60)

    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] Input not found: {args.input}")

    # Video info
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    process_frames = (min(int(args.seconds * fps), total_frames)
                      if args.seconds > 0 else total_frames)
    max_frames = process_frames if args.seconds > 0 else 0

    # Output paths
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or os.path.join(_PROJECT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    hand_video_path = os.path.join(out_dir, f"{base}_hands.mp4")
    body_video_path = os.path.join(out_dir, f"{base}_body.mp4")
    masks_json_path = os.path.join(out_dir, f"{base}_masks_meta.json")

    print(f"  Input:   {args.input}  ({W}x{H}, {fps:.1f} FPS, {total_frames} frames)")
    print(f"  Process: {process_frames} frames ({process_frames / fps:.1f}s)")
    print(f"  Prompt:  \"{args.prompt}\"")
    print(f"  Output:  {out_dir}/")

    # ── Step 1: SAM3 image-mode hand detection ────────────────────
    print(f"\n[1/5] SAM3 image mode — Hand detection on frame {args.prompt_frame}...")
    hand_masks_initial = detect_hands_sam3_image(
        args.input, args.prompt_frame, args.prompt,
        confidence_threshold=args.thresh, max_hands=2,
    )

    if not hand_masks_initial:
        print("       ✗ No hands detected — exiting.")
        sys.exit(1)
    print(f"       ✓ {len(hand_masks_initial)} hand(s) detected")

    # ── Step 2: Extract frames for SAM2 ───────────────────────────
    print(f"\n[2/5] Extracting frames for SAM2...")
    frames_dir = os.path.splitext(args.input)[0] + "_frames_tmp"
    extract_frames_jpg(args.input, frames_dir,
                       max_frames=max_frames if max_frames > 0 else 0)

    # ── Step 3: SAM2 propagation for hands ────────────────────────
    print(f"\n[3/5] SAM2 — Propagating hand masks through video...")
    hand_mask_dicts = propagate_masks_sam2(
        frames_dir, args.prompt_frame, hand_masks_initial,
        obj_id_start=10,  # offset to avoid collision with body obj_id=1
    )

    for i, md in enumerate(hand_mask_dicts):
        print(f"       Hand {i}: {len(md)} frames with mask")

    # ── Step 4: SAM2 body segmentation ────────────────────────────
    print(f"\n[4/5] SAM2 — Body segmentation (point at lower-center)...")
    body_masks = segment_body_sam2(frames_dir, args.prompt_frame, W, H)
    print(f"       Body: {len(body_masks)} frames")

    # Clean up temp frames
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"       Cleaned up {frames_dir}")

    # ── Step 5: Assign L/R and render ─────────────────────────────
    print(f"\n[5/5] Assigning L/R and rendering videos...")
    left_hand_masks, right_hand_masks = assign_left_right(hand_mask_dicts, W)
    print(f"       Left Hand:  {len(left_hand_masks)} frames")
    print(f"       Right Hand: {len(right_hand_masks)} frames")

    render_hand_video(args.input, left_hand_masks, right_hand_masks,
                      hand_video_path, fps, W, H)
    render_body_video(args.input, body_masks, left_hand_masks, right_hand_masks,
                      body_video_path, fps, W, H, dilate_px=args.hand_dilate)

    # ── Step 6: Export Masks & BBoxes ─────────────────────────────
    print(f"\n[6/6] Exporting masks and bounding boxes...")
    masks_out_dir = os.path.join(out_dir, f"{base}_masks")
    os.makedirs(masks_out_dir, exist_ok=True)
    bboxes_json_path = os.path.join(out_dir, f"{base}_bboxes.json")
    
    bbox_frames = {}
    for fidx in range(process_frames):
        lm = left_hand_masks.get(fidx)
        rm = right_hand_masks.get(fidx)
        
        # Save .npy mask stack: [Left, Right, Dummy, Dummy]
        # (Dummy added to be compatible with apache_hand_skeleton expecting 4 channels)
        stack = []
        stack.append(lm if lm is not None else np.zeros((H, W), dtype=bool))
        stack.append(rm if rm is not None else np.zeros((H, W), dtype=bool))
        stack.append(np.zeros((H, W), dtype=bool))
        stack.append(np.zeros((H, W), dtype=bool))
        np.save(os.path.join(masks_out_dir, f"frame_{fidx:06d}.npy"), np.stack(stack))
        
        # Save bboxes
        frame_bboxes = []
        l_bbox = mask_to_bbox(lm)
        if l_bbox:
            frame_bboxes.append({"bbox": l_bbox, "obj_id": 0})
        r_bbox = mask_to_bbox(rm)
        if r_bbox:
            frame_bboxes.append({"bbox": r_bbox, "obj_id": 1})
            
        bbox_frames[str(fidx)] = frame_bboxes
        
    with open(bboxes_json_path, "w") as f:
        json.dump({"width": W, "height": H, "frames": bbox_frames}, f, indent=2)
    print(f"       Exported masks to {masks_out_dir}/")
    print(f"       Exported bboxes to {bboxes_json_path}")

    # Save metadata
    meta = {
        "version": "0.1",
        "pipeline": "SAM3 image + SAM2 video",
        "input": args.input,
        "hand_prompt": args.prompt,
        "frames_processed": process_frames,
        "left_hand_frames": len(left_hand_masks),
        "right_hand_frames": len(right_hand_masks),
        "body_frames": len(body_masks),
        "hand_dilate_px": args.hand_dilate,
        "hand_video": hand_video_path,
        "body_video": body_video_path,
    }
    with open(masks_json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Hand video: {hand_video_path}")
    print(f"  Body video: {body_video_path}")
    print(f"  Bboxes:     {bboxes_json_path}")
    print(f"  Masks:      {masks_out_dir}/")
    print(f"  Metadata:   {masks_json_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

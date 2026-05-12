# Hand Pose Estimation — Egocentric Video Pipeline

Robust hand pose estimation for the main person in egocentric (first-person) videos.

## Architecture

```
Input Video
    │
    ▼
┌──────────────────┐
│  SAM3 Hand       │  Text-prompt detection + mask propagation
│  Tracking        │  → per-frame hand bounding boxes & masks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MediaPipe       │  Primary 21-joint hand landmark detector
│  HandLandmarker  │  (with rotation & half-crop fallbacks)
├──────────────────┤
│  ViTPose+        │  Fallback for gloved / occluded hands
│  (Huge)          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Post-processing │  Handedness correction, temporal smoothing,
│  & Export        │  JSON export, overlay video rendering
└──────────────────┘
```

## Quick Start

```bash
# 1. Set up environment
bash scripts/setup_env.sh

# 2. Place input video
cp your_video.mp4 data/input.mp4

# 3. Run pipeline (see version-specific instructions in versions/)
python src/track_hands.py data/input.mp4
python src/estimate_skeleton.py data/input.mp4
```

## Version History

| Version | Date       | Highlights |
|---------|------------|------------|
| v0.0    | 2026-05-12 | Project scaffold, workflow tooling |

See [`versions/`](versions/) for detailed changelogs.

## Development Workflow

### Version Management
```bash
# Create a new version
bash scripts/new_version.sh 0.1 "SAM3 Hand Tracking"

# Commit and push current work
bash scripts/commit_and_push.sh "Added hand tracking module"
```

### Goal-Driven Development
Define goals in `goals/current_goals.yaml` for Claude to execute:
```bash
# View current goals
make goals

# After Claude completes work, commit
make commit
```

See [`goals/`](goals/) for the goal specification format.

## Constraints

- **License**: Apache 2.0 components only (no MANO, no HaPTIC)
- **Hardware**: Single GPU (RTX 4090)
- **Focus**: Egocentric hand pose — main person only, no bystanders

## Project Structure

```
hand_pose_ag/
├── configs/          Pipeline configuration (YAML)
├── data/             Input videos & data (git-ignored)
├── goals/            Claude goal specifications
├── outputs/          Pipeline outputs (git-ignored)
├── scripts/          Workflow scripts (commit, version, setup)
├── src/              Pipeline source code
└── versions/         Version changelogs
```

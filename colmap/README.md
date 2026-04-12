# COLMAP Camera Calibration Results

## Directory Structure

```
colmap/
├── README.md                   # This file
├── colmap_workspace/           # Round 1 results
│   ├── database.db             # COLMAP DB (round 1)
│   ├── images_flat/            # All frames flat (1680 JPG)
│   ├── images_selected/        # Selected 168 frames
│   ├── images/                 # Camera subdirectories (empty)
│   ├── sparse/                 # Sparse reconstructions
│   │   ├── 0/                  # Recon 0 (2 images)
│   │   ├── 1/                  # Recon 1 (103 images) ← Main
│   │   └── 2/                  # Recon 2 (10 images)
│   ├── colmap_results.json     # Detailed results
│   └── colmap_summary.json     # Summary
│
└── round2/                     # Round 2 results (FINAL)
    ├── database.db             # COLMAP DB (round 2)
    ├── images/                 # Camera-per-folder structure
    │   ├── cam1/               # 48 frames (portrait 1080x1920)
    │   ├── cam2/               # 48 frames (landscape 1920x1080)
    │   ├── cam3/               # 48 frames
    │   ├── cam4/               # 48 frames
    │   ├── cam5/               # 48 frames
    │   ├── cam6/               # 48 frames
    │   └── cam7/               # 48 frames
    ├── sparse/
    │   ├── 0/                  # Small recon (4 images)
    │   └── 1/                  # MAIN: 336/336 images, 7 cameras
    └── analysis.json           # Detailed analysis
```

## Key Results

| Metric | Round 1 | Round 2 |
|--------|---------|---------|
| Camera Model | OPENCV (8 params) | SIMPLE_RADIAL (4 params) |
| Camera Sharing | None (per-image) | Per-folder (7 cameras) |
| Input Images | 168 | 336 |
| Registered | 103 (61.3%) | **336 (100%)** |
| 3D Points | 6,218 | 7,005 |
| Reproj Error (median) | 0.40 px | 0.75 px |
| Cameras | 103 (per-image) | **7 (physical)** |

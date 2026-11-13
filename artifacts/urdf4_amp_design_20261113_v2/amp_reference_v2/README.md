URDF4 AMP reference V2: faster/higher-clearance diagonal motion candidates.
12 clips × 2500 frames, 50Hz, 46D features. New physical controller design; no policy checkpoint used.
Actual forward speed ~0.290m/s; per-foot P95 clearance ~27–35mm.
V1 files are preserved byte-for-byte. V2 is only MuJoCo-screened; amp_training_approved=false until matching Isaac Gym AMP validation. Read manifest.json and the parent REPORT.md.
Motor limits are unchanged. Higher clearance increases body vertical motion; no thermal/real-robot approval is implied.

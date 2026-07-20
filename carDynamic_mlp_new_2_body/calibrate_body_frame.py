"""
Run this against your REAL training data to determine which of the 4
possible (forward, lateral) sign/axis conventions actually matches your
car_linear_velocity_x / car_linear_velocity_y channels.

Usage:
    python calibrate_body_frame.py

Reads data_dir + data_source straight out of config.json, so run it from
the same directory as main.py/config.json.
"""
import json
import numpy as np
import pandas as pd
from utils import STATE_COLS, _add_derived_columns

IDX_VX = STATE_COLS.index('car_linear_velocity_x')
IDX_VY = STATE_COLS.index('car_linear_velocity_y')

# The 4 candidate proper-rotation conventions (all det=+1, all use the
# state's sin_theta/cos_theta directly -- they only differ in which world
# axis is treated as the theta=0 forward direction and which rotation
# direction is "left").
CANDIDATES = {
    'A: fwd=cos*dx+sin*dy, lat=-sin*dx+cos*dy': lambda dx, dy, s, c: (c*dx + s*dy, -s*dx + c*dy),
    'B: fwd=sin*dx+cos*dy, lat=-cos*dx+sin*dy': lambda dx, dy, s, c: (s*dx + c*dy, -c*dx + s*dy),
    'C: fwd=-cos*dx-sin*dy, lat=sin*dx-cos*dy': lambda dx, dy, s, c: (-c*dx - s*dy, s*dx - c*dy),
    'D: fwd=-sin*dx-cos*dy, lat=cos*dx-sin*dy': lambda dx, dy, s, c: (-s*dx - c*dy, c*dx - s*dy),
}

with open('config.json') as f:
    config = json.load(f)

all_dx, all_dy, all_s, all_c, all_vx, all_vy, all_dt = [], [], [], [], [], [], []

for fname in config['data_source']:
    data = pd.read_csv(f"{config['data_dir']}/{fname}")
    data = _add_derived_columns(data)
    state = data[STATE_COLS].values
    ts = data['timestamp'].values

    dx = np.diff(state[:, 0])
    dy = np.diff(state[:, 1])
    dt = np.diff(ts)
    # drop episode-reset jumps (reuse the same 0.14 heuristic as utils.py)
    jump = np.hypot(dx, dy)
    keep = jump < 0.14
    dx, dy, dt = dx[keep], dy[keep], dt[keep]
    s = state[:-1, STATE_COLS.index('sin_theta')][keep]
    c = state[:-1, STATE_COLS.index('cos_theta')][keep]
    vx = ((state[:-1, IDX_VX] + state[1:, IDX_VX]) / 2)[keep]
    vy = ((state[:-1, IDX_VY] + state[1:, IDX_VY]) / 2)[keep]

    all_dx.append(dx); all_dy.append(dy); all_s.append(s); all_c.append(c)
    all_vx.append(vx); all_vy.append(vy); all_dt.append(dt)

dx = np.concatenate(all_dx); dy = np.concatenate(all_dy)
s = np.concatenate(all_s); c = np.concatenate(all_c)
vx = np.concatenate(all_vx); vy = np.concatenate(all_vy)
dt = np.concatenate(all_dt)

expect_forward = vy * dt   # trapezoidal target used by kinematic_physics_loss
expect_lateral = vx * dt

print(f"{'candidate':<45} {'fwd_corr':>9} {'fwd_rmse':>10} {'lat_corr':>9} {'lat_rmse':>10}  score")
best_name, best_score = None, -np.inf
for name, fn in CANDIDATES.items():
    fwd, lat = fn(dx, dy, s, c)
    fwd_corr = np.corrcoef(fwd, expect_forward)[0, 1]
    lat_corr = np.corrcoef(lat, expect_lateral)[0, 1]
    fwd_rmse = np.sqrt(np.mean((fwd - expect_forward) ** 2))
    lat_rmse = np.sqrt(np.mean((lat - expect_lateral) ** 2))
    score = fwd_corr + lat_corr
    print(f"{name:<45} {fwd_corr:>9.4f} {fwd_rmse:>10.5f} {lat_corr:>9.4f} {lat_rmse:>10.5f}  {score:.4f}")
    if score > best_score:
        best_score, best_name = score, name

print(f"\nBest match: {best_name}")
print("Use this formula in utils.py's state_delta_body_frame / apply_body_frame_delta.")
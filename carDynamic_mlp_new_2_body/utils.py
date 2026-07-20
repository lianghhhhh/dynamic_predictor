import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# CONTROL_COLS: truly exogenous quantities -- the commands you send. These
# are the only things known ahead of time / available at every step of a
# closed-loop rollout without the model having to predict them.
CONTROL_COLS = [
    'effort_command_front_left', 'effort_command_front_right',
    # 'effort_command_rear_left', 'effort_command_rear_right',
]

# STATE_COLS: the dynamic state of the car, including the wheels. Wheel
# angle/velocity are measured RESPONSES to the effort command (actuator
# dynamics: lag, slip, friction), not commands themselves, so the
# model has
# to predict them just like position/velocity/heading. This is what the
# model predicts (as a delta) and it is exactly what gets fed back in as
# "current state" for the next rollout step, so input-state and
# output-state dimensions match (7).
STATE_COLS = [
    'car_position_x', 'car_position_y',
    'sin_theta', 'cos_theta',
    'car_linear_velocity_x', 'car_linear_velocity_y', 'car_angular_velocity_z',
    # 'sin_angle_fl', 'cos_angle_fl', 'sin_angle_fr', 'cos_angle_fr',
    # 'sin_angle_rl', 'cos_angle_rl', 'sin_angle_rr', 'cos_angle_rr',
    # 'front_left_velocity', 'front_right_velocity',
    # 'rear_left_velocity', 'rear_right_velocity',
]

# (sin_index, cos_index) pairs within STATE_COLS that need renormalizing
# back onto the unit circle after a predicted delta is added during rollout.
SINCOS_PAIRS = [
    (STATE_COLS.index('sin_theta'), STATE_COLS.index('cos_theta')),
    # (STATE_COLS.index('sin_angle_fl'), STATE_COLS.index('cos_angle_fl')),
    # (STATE_COLS.index('sin_angle_fr'), STATE_COLS.index('cos_angle_fr')),
    # (STATE_COLS.index('sin_angle_rl'), STATE_COLS.index('cos_angle_rl')),
    # (STATE_COLS.index('sin_angle_rr'), STATE_COLS.index('cos_angle_rr')),
]

# Absolute x/y position is meaningless as a *feature* (dynamics don't depend
# on where in the world you are -- feeding it in just invites overfitting to
# the coordinate ranges seen in training). It still lives in STATE_COLS
# because the model needs to predict its delta and rollout needs to track
# absolute position for comparison/plotting -- it's just excluded from the
# "absolute state" block of the model's input features.
POSITION_COLS = ['car_position_x', 'car_position_y']
ABS_STATE_MASK = np.array([col not in POSITION_COLS for col in STATE_COLS])
N_ABS_STATE = int(ABS_STATE_MASK.sum())

LOOKAHEAD = 1
SEQ_LENGTH = 10
RESET_JUMP_THRESHOLD = 0.14  # position jump bigger than this => treat as episode reset

# Indices of the position/heading channels within STATE_COLS, used to
# convert position deltas between world frame and the car's body frame
# (forward/lateral). car_position_x/y themselves stay world-frame absolute
# values -- only their *deltas* (model input history + prediction target)
# get expressed in body frame, since forward/lateral motion is what the
# actuators actually control and generalizes much better than world dx/dy.
POS_X_IDX = STATE_COLS.index('car_position_x')
POS_Y_IDX = STATE_COLS.index('car_position_y')
SIN_IDX = STATE_COLS.index('sin_theta')
COS_IDX = STATE_COLS.index('cos_theta')

# Which world axis is "forward" at heading==0, and which rotation sense is
# "left" -- run calibrate_body_frame.py against your real training data to
# determine this empirically (it checks correlation against
# car_linear_velocity_x/y, which is the ground truth for what your sim
# actually calls forward/lateral). Getting this wrong doesn't break the
# training math (it's still a proper, invertible rotation either way) but
# it DOES make the kinematic_physics_loss (weight_xy) fight the model
# instead of helping it, since that loss compares against vx/vy directly.
#   'A' -> forward = cos*dx + sin*dy,  lateral = -sin*dx + cos*dy
#   'B' -> forward = sin*dx + cos*dy,  lateral = -cos*dx + sin*dy
#   'C' -> forward = -cos*dx - sin*dy, lateral = sin*dx - cos*dy
#   'D' -> forward = -sin*dx - cos*dy, lateral = cos*dx - sin*dy
BODY_FRAME_CONVENTION = 'B'  # <-- set this from calibrate_body_frame.py's output


def _rotate_world_to_body(dx, dy, sin_h, cos_h):
    if BODY_FRAME_CONVENTION == 'A':
        return cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy
    elif BODY_FRAME_CONVENTION == 'B':
        return sin_h * dx + cos_h * dy, -cos_h * dx + sin_h * dy
    elif BODY_FRAME_CONVENTION == 'C':
        return -cos_h * dx - sin_h * dy, sin_h * dx - cos_h * dy
    elif BODY_FRAME_CONVENTION == 'D':
        return -sin_h * dx - cos_h * dy, cos_h * dx - sin_h * dy
    else:
        raise ValueError(f"Unknown BODY_FRAME_CONVENTION: {BODY_FRAME_CONVENTION}")


def _rotate_body_to_world(forward, lateral, sin_h, cos_h):
    if BODY_FRAME_CONVENTION == 'A':
        return cos_h * forward - sin_h * lateral, sin_h * forward + cos_h * lateral
    elif BODY_FRAME_CONVENTION == 'B':
        return sin_h * forward - cos_h * lateral, cos_h * forward + sin_h * lateral
    elif BODY_FRAME_CONVENTION == 'C':
        return -cos_h * forward + sin_h * lateral, -sin_h * forward - cos_h * lateral
    elif BODY_FRAME_CONVENTION == 'D':
        return -sin_h * forward + cos_h * lateral, -cos_h * forward - sin_h * lateral
    else:
        raise ValueError(f"Unknown BODY_FRAME_CONVENTION: {BODY_FRAME_CONVENTION}")


def state_delta_body_frame(state_from, state_to):
    """Delta between two absolute (world-frame) state rows, with the
    position component rotated into the body frame of `state_from`
    (forward = along heading, lateral = to the left of heading) instead of
    raw world-frame dx/dy. Every other channel (sin/cos heading, linear
    velocities, angular velocity) is a plain difference, same as before.

    Works on a single row (1D, shape (N_STATE,)) or a batch of rows (shape
    (..., N_STATE)) as long as the last axis is the STATE_COLS axis.
    """
    delta = np.asarray(state_to) - np.asarray(state_from)
    sin_h = state_from[..., SIN_IDX]
    cos_h = state_from[..., COS_IDX]
    dx = delta[..., POS_X_IDX]
    dy = delta[..., POS_Y_IDX]
    forward, lateral = _rotate_world_to_body(dx, dy, sin_h, cos_h)
    delta = delta.copy()
    delta[..., POS_X_IDX] = forward
    delta[..., POS_Y_IDX] = lateral
    return delta


def apply_body_frame_delta(state_from, delta):
    """Inverse of state_delta_body_frame: given an absolute (world-frame)
    state row and a delta whose position component is (forward, lateral)
    expressed in the body frame of `state_from`, return the resulting
    absolute world-frame state. Non-position channels are added directly.
    Does NOT renormalize sin/cos -- call renormalize_sincos on the result.
    """
    sin_h = state_from[..., SIN_IDX]
    cos_h = state_from[..., COS_IDX]
    forward = delta[..., POS_X_IDX]
    lateral = delta[..., POS_Y_IDX]
    dx, dy = _rotate_body_to_world(forward, lateral, sin_h, cos_h)

    next_state = state_from + delta  # correct for sin/cos/vx/vy/wz channels
    next_state[..., POS_X_IDX] = state_from[..., POS_X_IDX] + dx
    next_state[..., POS_Y_IDX] = state_from[..., POS_Y_IDX] + dy
    return next_state


def compute_angle(qx, qy, qz, qw):
    """Yaw angle from quaternion (planar car, so only yaw matters)."""
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def _add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    # data['sin_angle_fl'] = np.sin(data['front_left_angle'])
    # data['cos_angle_fl'] = np.cos(data['front_left_angle'])
    # data['sin_angle_fr'] = np.sin(data['front_right_angle'])
    # data['cos_angle_fr'] = np.cos(data['front_right_angle'])
    # data['sin_angle_rl'] = np.sin(data['rear_left_angle'])
    # data['cos_angle_rl'] = np.cos(data['rear_left_angle'])
    # data['sin_angle_rr'] = np.sin(data['rear_right_angle'])
    # data['cos_angle_rr'] = np.cos(data['rear_right_angle'])

    car_angle = compute_angle(data['car_orientation_x'], data['car_orientation_y'],
                               data['car_orientation_z'], data['car_orientation_w'])

    # flip angle
    car_angle = -car_angle
    data['sin_theta'] = np.sin(car_angle)
    data['cos_theta'] = np.cos(car_angle)

    # flip velocity
    data['car_linear_velocity_x'] = -data['car_linear_velocity_x']
    data['car_linear_velocity_y'] = -data['car_linear_velocity_y']
    data['car_angular_velocity_z'] = -data['car_angular_velocity_z']

    return data


def _build_windows(data: pd.DataFrame, source_name: str, seq_length=SEQ_LENGTH, lookahead=LOOKAHEAD):
    """Slide a window over one (already-processed) dataframe and emit
    (input_sequence, output_delta) pairs. Shared by train/test loaders."""
    inputs, outputs = [], []
    count = len(data) - lookahead

    control_vals = data[CONTROL_COLS].values
    state_vals = data[STATE_COLS].values

    for i in range(count - seq_length):
        start_pos = state_vals[i + seq_length - 1, 0:2]
        end_pos = state_vals[i + seq_length, 0:2]
        if np.linalg.norm(end_pos - start_pos) > RESET_JUMP_THRESHOLD:
            print(f"{source_name}: skipping window at {i}, looks like an episode reset "
                  f"({start_pos} -> {end_pos})")
            continue

        window = []
        for j in range(seq_length):
            idx = i + j
            prev_idx = idx - 1 if idx - 1 >= 0 else idx  # first element: no previous, delta=0
            delta_state = state_delta_body_frame(state_vals[prev_idx], state_vals[idx])
            window.append(np.concatenate([control_vals[idx], state_vals[idx][ABS_STATE_MASK], delta_state]))
        inputs.append(window)

        cur = state_vals[i + seq_length - 1]
        nxt = state_vals[i + seq_length]
        outputs.append(state_delta_body_frame(cur, nxt))

    return inputs, outputs


def renormalize_sincos(state_row):
    """Project all sin/cos pairs in a state vector back onto the unit
    circle. Needed after adding a predicted delta, since nothing forces
    sin^2+cos^2==1 to hold for an unconstrained regression output."""
    for sin_idx, cos_idx in SINCOS_PAIRS:
        norm = np.hypot(state_row[sin_idx], state_row[cos_idx])
        if norm > 1e-6:
            state_row[sin_idx] /= norm
            state_row[cos_idx] /= norm
    return state_row


def assemble_window(controls, abs_state, delta_state, t, seq_length):
    """Build the seq_length x (N_CONTROL+N_ABS_STATE+N_STATE) raw input
    window ending at index t (inclusive) of the given per-timestep arrays.
    Absolute x/y position is excluded from the absolute-state block
    (ABS_STATE_MASK) -- only its delta is fed in. Shared by training
    (multi-step rollout loss) and evaluation, so the two stay consistent."""
    rows = []
    for j in range(t - seq_length + 1, t + 1):
        rows.append(np.concatenate([controls[j], abs_state[j][ABS_STATE_MASK], delta_state[j]]))
    return np.array(rows)


def make_delta_buffer(abs_state_window):
    """Given a seq_length x N_STATE absolute-state window, compute the
    consecutive deltas the same way training data does (first row gets
    zero delta, since there's no earlier sample). Position channels come
    back as body-frame (forward/lateral) deltas, not world dx/dy."""
    delta = np.zeros_like(abs_state_window)
    for j in range(1, len(abs_state_window)):
        delta[j] = state_delta_body_frame(abs_state_window[j - 1], abs_state_window[j])
    return delta



def get_data(data_dir, data_source, seq_length=SEQ_LENGTH, lookahead=LOOKAHEAD):
    all_inputs, all_outputs = [], []
    for file in data_source:
        data = pd.read_csv(os.path.join(data_dir, file))
        data = _add_derived_columns(data)
        ins, outs = _build_windows(data, file, seq_length, lookahead)
        all_inputs.extend(ins)
        all_outputs.extend(outs)
    return all_inputs, all_outputs


def get_test_data(data_dir, test_data_source, seq_length=SEQ_LENGTH, lookahead=LOOKAHEAD):
    """Kept for backward-compat one-step-ahead evaluation. Also returns the
    raw absolute state at the *end* of each input window (the anchor state),
    which is what you need to turn a predicted delta back into an absolute
    state for plotting."""
    data = pd.read_csv(os.path.join(data_dir, test_data_source))
    data = _add_derived_columns(data)
    inputs, outputs = _build_windows(data, test_data_source, seq_length, lookahead)

    state_vals = data[STATE_COLS].values
    count = len(data) - lookahead
    anchor_states = []
    kept_i = []
    for i in range(count - seq_length):
        start_pos = state_vals[i + seq_length - 1, 0:2]
        end_pos = state_vals[i + seq_length, 0:2]
        if np.linalg.norm(end_pos - start_pos) > RESET_JUMP_THRESHOLD:
            continue
        anchor_states.append(state_vals[i + seq_length - 1])

    return inputs, outputs, anchor_states


def load_trajectory_segments(data_dir, file, seq_length=SEQ_LENGTH):
    """Load one CSV and split it into continuous (no-reset) segments, each
    returned as raw per-timestep arrays. This is what closed-loop / rollout
    evaluation needs: real control inputs for every step, plus the true
    state trajectory to seed the rollout and to compare predictions against.

    Returns: list of dicts, each with:
        'controls': (T, len(CONTROL_COLS))
        'state':    (T, len(STATE_COLS))   -- ground truth
    Segments shorter than seq_length+2 are dropped (can't seed + roll at least 1 step).
    """
    data = pd.read_csv(os.path.join(data_dir, file))
    data = _add_derived_columns(data)
    control_vals = data[CONTROL_COLS].values
    state_vals = data[STATE_COLS].values

    segments = []
    seg_start = 0
    for t in range(1, len(data)):
        if np.linalg.norm(state_vals[t, 0:2] - state_vals[t - 1, 0:2]) > RESET_JUMP_THRESHOLD:
            if t - seg_start >= seq_length + 2:
                segments.append({
                    'controls': control_vals[seg_start:t],
                    'state': state_vals[seg_start:t],
                })
            seg_start = t
    if len(data) - seg_start >= seq_length + 2:
        segments.append({
            'controls': control_vals[seg_start:len(data)],
            'state': state_vals[seg_start:len(data)],
        })

    if not segments:
        print(f"{file}: no segment long enough for seq_length={seq_length}")
    return segments
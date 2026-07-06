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
    'effort_command_rear_left', 'effort_command_rear_right',
]

# STATE_COLS: the dynamic state of the car, including the wheels. Wheel
# angle/velocity are measured RESPONSES to the effort command (actuator
# dynamics: lag, slip, friction), not commands themselves, so the model has
# to predict them just like position/velocity/heading. This is what the
# model predicts (as a delta) and it is exactly what gets fed back in as
# "current state" for the next rollout step, so input-state and
# output-state dimensions match (19).
STATE_COLS = [
    'car_position_x', 'car_position_y',
    'sin_theta', 'cos_theta',
    'car_linear_velocity_x', 'car_linear_velocity_y', 'car_angular_velocity_z',
    'sin_angle_fl', 'cos_angle_fl', 'sin_angle_fr', 'cos_angle_fr',
    'sin_angle_rl', 'cos_angle_rl', 'sin_angle_rr', 'cos_angle_rr',
    'front_left_velocity', 'front_right_velocity',
    'rear_left_velocity', 'rear_right_velocity',
]

# (sin_index, cos_index) pairs within STATE_COLS that need renormalizing
# back onto the unit circle after a predicted delta is added during rollout.
SINCOS_PAIRS = [
    (STATE_COLS.index('sin_theta'), STATE_COLS.index('cos_theta')),
    (STATE_COLS.index('sin_angle_fl'), STATE_COLS.index('cos_angle_fl')),
    (STATE_COLS.index('sin_angle_fr'), STATE_COLS.index('cos_angle_fr')),
    (STATE_COLS.index('sin_angle_rl'), STATE_COLS.index('cos_angle_rl')),
    (STATE_COLS.index('sin_angle_rr'), STATE_COLS.index('cos_angle_rr')),
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
SEQ_LENGTH = 40
RESET_JUMP_THRESHOLD = 0.14  # position jump bigger than this => treat as episode reset


def compute_angle(qx, qy, qz, qw):
    """Yaw angle from quaternion (planar car, so only yaw matters)."""
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def _add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    data['sin_angle_fl'] = np.sin(data['front_left_angle'])
    data['cos_angle_fl'] = np.cos(data['front_left_angle'])
    data['sin_angle_fr'] = np.sin(data['front_right_angle'])
    data['cos_angle_fr'] = np.cos(data['front_right_angle'])
    data['sin_angle_rl'] = np.sin(data['rear_left_angle'])
    data['cos_angle_rl'] = np.cos(data['rear_left_angle'])
    data['sin_angle_rr'] = np.sin(data['rear_right_angle'])
    data['cos_angle_rr'] = np.cos(data['rear_right_angle'])

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
            delta_state = state_vals[idx] - state_vals[prev_idx]
            window.append(np.concatenate([control_vals[idx], state_vals[idx][ABS_STATE_MASK], delta_state]))
        inputs.append(window)

        cur = state_vals[i + seq_length - 1]
        nxt = state_vals[i + seq_length]
        outputs.append(nxt - cur)

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
    zero delta, since there's no earlier sample)."""
    delta = np.zeros_like(abs_state_window)
    for j in range(1, len(abs_state_window)):
        delta[j] = abs_state_window[j] - abs_state_window[j - 1]
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
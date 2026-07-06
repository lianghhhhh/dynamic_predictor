import os
import json
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from carDynamicModel import CarDynamicModel
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import (get_data, load_trajectory_segments,
                    CONTROL_COLS, STATE_COLS, N_ABS_STATE,
                    renormalize_sincos, assemble_window, make_delta_buffer, POSITION_COLS)

SEQ_LENGTH = 40
N_CONTROL = len(CONTROL_COLS)
N_STATE = len(STATE_COLS)
# index of [sin_theta, cos_theta] inside STATE_COLS, used for the heading-error metric
SIN_IDX, COS_IDX = STATE_COLS.index('sin_theta'), STATE_COLS.index('cos_theta')


class TorchStandardScaler(nn.Module):
    """Unscales tensors back to physical units while keeping the PyTorch computation graph intact."""
    def __init__(self, sklearn_scaler, device):
        super().__init__()
        self.mean = torch.tensor(sklearn_scaler.mean_, dtype=torch.float32, device=device)
        self.scale = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32, device=device)

    def inverse_transform(self, scaled_tensor):
        return (scaled_tensor * self.scale) + self.mean


def kinematic_physics_loss(pred_delta_physical, cur_abs_physical, state_cols, dt=0.05):
    idx_vx = state_cols.index('car_linear_velocity_x')
    idx_wz = state_cols.index('car_angular_velocity_z')
    idx_sin = state_cols.index('sin_theta')
    idx_cos = state_cols.index('cos_theta')

    loss_xy = torch.tensor(0.0, device=pred_delta_physical.device)

    next_vx = cur_abs_physical[..., idx_vx] + pred_delta_physical[..., idx_vx]
    wz = cur_abs_physical[..., idx_wz]
    
    # The hard mask works best. No penalty if spinning > 1.0 rad/s
    slip_mask = (torch.abs(wz) < 1.0).float()
    loss_slip = torch.mean((next_vx ** 2) * slip_mask)

    # Revert to Forward Euler integration
    expected_dtheta = wz * dt  
    
    cur_angle = torch.atan2(cur_abs_physical[..., idx_sin], cur_abs_physical[..., idx_cos])
    next_angle = torch.atan2(
        cur_abs_physical[..., idx_sin] + pred_delta_physical[..., idx_sin],
        cur_abs_physical[..., idx_cos] + pred_delta_physical[..., idx_cos]
    )
    
    pred_dtheta = torch.atan2(torch.sin(next_angle - cur_angle), torch.cos(next_angle - cur_angle))
    loss_theta = torch.mean((pred_dtheta - expected_dtheta) ** 2)

    return loss_xy, loss_theta, loss_slip


def _sample_multistep_batch(segments, batch_size, seq_length, k_steps, rng):
    """Randomly pick `batch_size` starting points across training segments,
    long enough to seed a seq_length window and then roll forward k_steps.
    Segments are weighted by length so longer files contribute proportionally
    more samples."""
    lengths = np.array([len(s['state']) for s in segments], dtype=float)
    weights = np.maximum(lengths - seq_length - k_steps, 0)
    if weights.sum() <= 0:
        raise ValueError("No training segment is long enough for the requested "
                          f"seq_length={seq_length} + k_steps={k_steps}.")
    weights /= weights.sum()

    controls_buf, abs_buf, delta_buf = [], [], []
    future_controls, future_abs = [], []
    for _ in range(batch_size):
        seg = segments[rng.choice(len(segments), p=weights)]
        max_start = len(seg['state']) - seq_length - k_steps
        start = rng.integers(0, max_start + 1)

        abs_window = seg['state'][start:start + seq_length].astype(float)
        controls_buf.append(seg['controls'][start:start + seq_length])
        abs_buf.append(abs_window)
        delta_buf.append(make_delta_buffer(abs_window))
        future_controls.append(seg['controls'][start + seq_length: start + seq_length + k_steps])
        future_abs.append(seg['state'][start + seq_length: start + seq_length + k_steps])

    return (np.array(controls_buf), np.array(abs_buf), np.array(delta_buf),
            np.array(future_controls), np.array(future_abs))


def _multistep_training_step(model, segments, scaler_in, scaler_out, torch_scaler_out, optimizer, device,
                              criterion, batch_size, k_steps, seq_length, grad_clip, rng, physics_weight=100):
    
    controls_buf, abs_buf, delta_buf, future_controls, future_abs = _sample_multistep_batch(
        segments, batch_size, seq_length, k_steps, rng)
    
    cur_abs = abs_buf[:, -1, :].copy()
    feat_dim = N_CONTROL + N_ABS_STATE + N_STATE

    total_loss = torch.zeros((), device=device)
    warmup_steps = k_steps // 4 
    
    for step in range(k_steps):
        windows = np.stack([
            assemble_window(controls_buf[b], abs_buf[b], delta_buf[b], t=seq_length - 1, seq_length=seq_length)
            for b in range(batch_size)
        ])
        windows_scaled = scaler_in.transform(windows.reshape(-1, feat_dim)).reshape(windows.shape)
        x = torch.tensor(windows_scaled, dtype=torch.float32, device=device)

        pred_delta_scaled = model(x)  # (B, N_STATE), gradient-attached

        target_delta = future_abs[:, step, :] - cur_abs
        target_delta_scaled = scaler_out.transform(target_delta)
        target_tensor = torch.tensor(target_delta_scaled, dtype=torch.float32, device=device)

        if step >= warmup_steps:
            # --- 1. Data Loss ---
            data_loss = criterion(pred_delta_scaled, target_tensor)
            
            # Unscale the prediction 
            pred_delta_phys = torch_scaler_out.inverse_transform(pred_delta_scaled)
            cur_abs_tensor = torch.tensor(cur_abs, dtype=torch.float32, device=device)
            
            # Get the three separate physics losses
            loss_xy, loss_theta, loss_slip = kinematic_physics_loss(pred_delta_phys, cur_abs_tensor, STATE_COLS, dt=0.05)
            
            # --- Tune these weights independently ---
            # If straight lines wobble, lower weight_theta.
            # If turning is too fast/drifting, increase weight_slip.
            weight_xy = 0.0 
            weight_theta = 20.0 
            weight_slip = 50.0  
            
            phys_loss_total = (weight_xy * loss_xy) + (weight_theta * loss_theta) + (weight_slip * loss_slip)
            
            total_loss = total_loss + data_loss + phys_loss_total

        # roll forward using the model's OWN prediction (detached)
        pred_delta = scaler_out.inverse_transform(pred_delta_scaled.detach().cpu().numpy())
        next_abs = cur_abs + pred_delta
        for b in range(batch_size):
            next_abs[b] = renormalize_sincos(next_abs[b])
        new_delta = next_abs - cur_abs

        abs_buf = np.concatenate([abs_buf[:, 1:, :], next_abs[:, None, :]], axis=1)
        delta_buf = np.concatenate([delta_buf[:, 1:, :], new_delta[:, None, :]], axis=1)
        controls_buf = np.concatenate(
            [controls_buf[:, 1:, :], future_controls[:, step, :][:, None, :]], axis=1)
        cur_abs = next_abs

    avg_loss = total_loss / (k_steps - warmup_steps)
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return avg_loss.item()


def _rollout_validation_score(model, segments, scaler_in, scaler_out, device,
                               seq_length, horizon=10, max_segments=6, starts_per_segment=3):
    """Compute a cheap closed-loop score for checkpoint selection.

    Lower is better. This measures mean XY position error after the model
    feeds its own predictions back into the input window.
    """
    if not segments:
        return float('inf')

    rng = np.random.default_rng(0)
    segment_indices = np.arange(len(segments))
    rng.shuffle(segment_indices)
    segment_indices = segment_indices[:min(max_segments, len(segment_indices))]

    pos_errors = []
    model.eval()
    with torch.no_grad():
        for seg_idx in segment_indices:
            seg = segments[seg_idx]
            controls = seg['controls']
            true_state = seg['state']
            T = len(true_state)
            if T < seq_length + 2:
                continue

            stride = max(1, horizon // 2)
            starts = np.arange(0, T - seq_length - 1, stride)
            if len(starts) == 0:
                continue
            rng.shuffle(starts)

            for start in starts[:starts_per_segment]:
                roll_len = min(horizon, T - (start + seq_length))
                if roll_len <= 0:
                    continue

                abs_state = true_state[start:start + seq_length].copy().astype(float)
                delta_state = make_delta_buffer(abs_state)
                cur_controls = controls[start:start + seq_length].copy()

                for h in range(roll_len):
                    window = assemble_window(cur_controls, abs_state, delta_state,
                                              t=seq_length - 1, seq_length=seq_length)
                    window_scaled = scaler_in.transform(window)
                    x = torch.tensor(window_scaled, dtype=torch.float32, device=device).unsqueeze(0)

                    pred_delta_scaled = model(x).cpu().numpy()[0]
                    pred_delta = scaler_out.inverse_transform(pred_delta_scaled.reshape(1, -1))[0]

                    cur_abs = abs_state[-1]
                    pred_abs = renormalize_sincos(cur_abs + pred_delta)
                    true_abs = true_state[start + seq_length + h]
                    pos_errors.append(np.linalg.norm(pred_abs[0:2] - true_abs[0:2]))

                    next_control = controls[start + seq_length + h]
                    abs_state = np.vstack([abs_state[1:], pred_abs])
                    new_delta = pred_abs - cur_abs
                    delta_state = np.vstack([delta_state[1:], new_delta])
                    cur_controls = np.vstack([cur_controls[1:], next_control])

    return float(np.mean(pos_errors)) if pos_errors else float('inf')


def train_model(model, train_loader, val_loader, epochs, learning_rate, writer, device,
                 checkpoint_dir='checkpoints', patience=15, grad_clip=0.5, weight_decay=1e-5,
                 noise_std_max=0.05, noise_warmup_epochs=10,
                 multistep_segments=None, scaler_in=None, scaler_out=None,
                 rollout_val_segments=None, rollout_val_horizon=10,
                 rollout_val_max_segments=6, rollout_val_starts_per_segment=3,
                 multistep_k=10, multistep_batches_per_epoch=20, multistep_batch_size=32,
                 multistep_seq_length=SEQ_LENGTH, multistep_weight=1.0, seed=0):
    """Trains with two complementary fixes for closed-loop drift:

    1. Noise injection: gaussian noise added to the state portion of the
       *input* (not the target) each training batch, ramped up over
       `noise_warmup_epochs`. Cheaply approximates "what my own slightly
       wrong predictions would look like," without needing real rollouts.
    2. Multi-step rollout training: if `multistep_segments` is given (list
       from utils.load_trajectory_segments), every epoch also runs
       `multistep_batches_per_epoch` batches where the model unrolls
       `multistep_k` steps using its own predictions, with loss compared to
       ground truth at every step. This directly optimizes for the thing
       closed-loop evaluation measures, instead of only one-step MSE.

    Either can be disabled: pass noise_std_max=0 to skip noise injection, or
    leave multistep_segments=None to skip multi-step training.
    """
    torch_scaler_out = TorchStandardScaler(scaler_out, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(seed)

    use_multistep = multistep_segments is not None and len(multistep_segments) > 0
    if use_multistep and (scaler_in is None or scaler_out is None):
        raise ValueError("multistep_segments was given but scaler_in/scaler_out were not.")

    use_rollout_val = rollout_val_segments is not None and len(rollout_val_segments) > 0
    if use_rollout_val and (scaler_in is None or scaler_out is None):
        raise ValueError("rollout_val_segments was given but scaler_in/scaler_out were not.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_checkpoint_metric = float('inf')
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(epochs):
        noise_std = noise_std_max * min(1.0, (epoch + 1) / max(1, noise_warmup_epochs))

        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if noise_std > 0:
                # only corrupt the state/delta-state portion of the input;
                # control commands (first N_CONTROL features) are genuinely
                # known exactly, so there's no reason to noise them
                noise = torch.zeros_like(inputs)
                noise[..., N_CONTROL:] = torch.randn_like(inputs[..., N_CONTROL:]) * noise_std
                inputs = inputs + noise

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        multistep_loss = float('nan')
        if use_multistep:
            ms_running = 0.0
            for _ in range(multistep_batches_per_epoch):
                ms_loss = _multistep_training_step(
                    model, multistep_segments, scaler_in, scaler_out, torch_scaler_out, optimizer, device,
                    criterion, multistep_batch_size, multistep_k, multistep_seq_length,
                    grad_clip, rng, physics_weight=100) # Tune physics_weight as needed
                ms_running += ms_loss
            multistep_loss = ms_running / multistep_batches_per_epoch

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_predictions = model(val_inputs)
                val_loss += criterion(val_predictions, val_targets).item()
        val_loss /= len(val_loader)

        rollout_val_loss = float('nan')
        checkpoint_metric = val_loss
        if use_rollout_val:
            rollout_val_loss = _rollout_validation_score(
                model, rollout_val_segments, scaler_in, scaler_out, device,
                seq_length=multistep_seq_length,
                horizon=rollout_val_horizon,
                max_segments=rollout_val_max_segments,
                starts_per_segment=rollout_val_starts_per_segment,
            )
            checkpoint_metric = rollout_val_loss

        scheduler.step(val_loss)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Multistep', multistep_loss, epoch)
        writer.add_scalar('Loss/RolloutValidation', rollout_val_loss, epoch)
        writer.add_scalar('NoiseStd', noise_std, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch + 1}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  "
              f"multistep={multistep_loss:.6f}  rollout_val={rollout_val_loss:.6f}  noise_std={noise_std:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # checkpoint by closed-loop improvement when available; otherwise fall back to one-step validation
        if checkpoint_metric < best_checkpoint_metric:
            best_checkpoint_metric = checkpoint_metric
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pth'))

        if epochs_without_improvement >= patience:
            print(f"No improvement for {patience} epochs, stopping early at epoch {epoch + 1}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


def _plot_trajectory_examples(model, segments, scaler_in, scaler_out, device,
                              seq_length, horizon, name, n_examples=6):
    """Overlay true vs predicted XY trajectories for a few rollout starts."""
    model.eval()
    examples = []
    rng = np.random.default_rng(0)

    # Candidate rollout starts across all segments.
    candidates = [
        (si, start)
        for si, seg in enumerate(segments)
        for start in range(0, len(seg['state']) - seq_length - horizon, max(1, horizon))
    ]
    rng.shuffle(candidates)

    with torch.no_grad():
        for si, start in candidates[:n_examples * 3]:
            if len(examples) >= n_examples:
                break

            seg = segments[si]
            T = len(seg['state'])
            if start + seq_length + horizon > T:
                continue

            controls = seg['controls']
            true_state = seg['state']

            abs_state = true_state[start:start + seq_length].copy().astype(float)
            delta_state = make_delta_buffer(abs_state)
            cur_controls = controls[start:start + seq_length].copy()

            true_xy = [true_state[start + seq_length - 1, :2].copy()]
            pred_xy = [abs_state[-1, :2].copy()]

            for h in range(horizon):
                window = assemble_window(cur_controls, abs_state, delta_state,
                                         t=seq_length - 1, seq_length=seq_length)
                window_scaled = scaler_in.transform(window)
                x = torch.tensor(window_scaled, dtype=torch.float32, device=device).unsqueeze(0)

                pred_delta_scaled = model(x).cpu().numpy()[0]
                pred_delta = scaler_out.inverse_transform(pred_delta_scaled.reshape(1, -1))[0]

                cur_abs = abs_state[-1]
                pred_abs = renormalize_sincos(cur_abs + pred_delta)

                real_idx = start + seq_length + h
                true_xy.append(true_state[real_idx, :2].copy())
                pred_xy.append(pred_abs[:2].copy())

                next_control = controls[real_idx]
                abs_state = np.vstack([abs_state[1:], pred_abs])
                new_delta = pred_abs - cur_abs
                delta_state = np.vstack([delta_state[1:], new_delta])
                cur_controls = np.vstack([cur_controls[1:], next_control])

            examples.append((np.array(true_xy), np.array(pred_xy)))

    if not examples:
        print("No trajectory examples to plot.")
        return

    cols = 3
    rows = (len(examples) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (true_xy, pred_xy) in enumerate(examples):
        ax = axs[idx // cols][idx % cols]
        ax.plot(true_xy[:, 0], true_xy[:, 1], 'b-o', markersize=3, label='true')
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'r--s', markersize=3, label='pred')
        ax.plot(true_xy[0, 0], true_xy[0, 1], 'k*', markersize=10, label='start')
        final_err = np.linalg.norm(pred_xy[-1] - true_xy[-1])
        ax.set_title(f'Rollout {idx + 1} (final err={final_err:.3f}m)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend(fontsize=7)
        ax.set_aspect('equal', adjustable='datalim')

    for idx in range(len(examples), rows * cols):
        axs[idx // cols][idx % cols].set_visible(False)

    plt.suptitle(f'True vs Predicted XY trajectories ({horizon}-step rollout)', fontsize=13)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


# ---------------------------------------------------------------------------
# Closed-loop (rollout) evaluation
# ---------------------------------------------------------------------------
def evaluate_model_rollout(model, data_dir, test_data_source, device, scaler_in, scaler_out,
                            seq_length=SEQ_LENGTH, horizon=50, name='rollout_results.png'):
    """Closed-loop evaluation: seed with `seq_length` real steps, then keep
    predicting one step ahead and FEEDING THE PREDICTION BACK IN as the next
    state (controls stay ground-truth, since they're exogenous commands).
    Error is tracked as a function of how many steps into the rollout you
    are, which is the standard way to see how a dynamics model drifts.
    """
    model.eval()
    segments = load_trajectory_segments(data_dir, test_data_source, seq_length)
    if not segments:
        print("No usable segments for rollout evaluation.")
        return

    # error[h] collects squared error at horizon-step h, across all rollouts
    pos_err = [[] for _ in range(horizon)]
    angle_err = [[] for _ in range(horizon)]
    vel_err = [[] for _ in range(horizon)]
    ang_vel_err = [[] for _ in range(horizon)]
    pos_true_x = [[] for _ in range(horizon)]
    pos_pred_x = [[] for _ in range(horizon)]
    pos_true_y = [[] for _ in range(horizon)]
    pos_pred_y = [[] for _ in range(horizon)]
    angle_true = [[] for _ in range(horizon)]
    angle_pred = [[] for _ in range(horizon)]
    heading_sin_true = [[] for _ in range(horizon)]
    heading_sin_pred = [[] for _ in range(horizon)]
    heading_cos_true = [[] for _ in range(horizon)]
    heading_cos_pred = [[] for _ in range(horizon)]
    vel_true_x = [[] for _ in range(horizon)]
    vel_pred_x = [[] for _ in range(horizon)]
    vel_true_y = [[] for _ in range(horizon)]
    vel_pred_y = [[] for _ in range(horizon)]
    ang_vel_true = [[] for _ in range(horizon)]
    ang_vel_pred = [[] for _ in range(horizon)]

    with torch.no_grad():
        for seg in segments:
            controls = seg['controls']
            true_state = seg['state']
            T = len(true_state)

            # slide rollout starting points across the segment so we don't
            # just get one rollout per file
            stride = max(1, horizon // 2)
            for start in range(0, T - seq_length - 1, stride):
                end_real_idx = min(start + seq_length + horizon, T - 1)
                roll_len = end_real_idx - (start + seq_length - 1)
                if roll_len < 2:
                    continue

                # rolling buffers: absolute state is ground truth for the
                # seed window, then becomes model predictions
                abs_state = true_state[start:start + seq_length].copy().astype(float)
                delta_state = make_delta_buffer(abs_state)

                cur_controls = controls[start:start + seq_length].copy()

                for h in range(roll_len - 1):
                    window = assemble_window(cur_controls, abs_state, delta_state,
                                              t=seq_length - 1, seq_length=seq_length)
                    window_scaled = scaler_in.transform(window)
                    x = torch.tensor(window_scaled, dtype=torch.float32, device=device).unsqueeze(0)

                    pred_delta_scaled = model(x).cpu().numpy()[0]
                    pred_delta = scaler_out.inverse_transform(pred_delta_scaled.reshape(1, -1))[0]

                    cur_abs = abs_state[-1]
                    pred_abs = renormalize_sincos(cur_abs + pred_delta)

                    real_idx = start + seq_length + h
                    true_abs = true_state[real_idx]

                    pe = np.linalg.norm(pred_abs[0:2] - true_abs[0:2])
                    pred_angle = np.arctan2(pred_abs[SIN_IDX], pred_abs[COS_IDX])
                    true_angle = np.arctan2(true_abs[SIN_IDX], true_abs[COS_IDX])
                    ae = abs(np.arctan2(np.sin(pred_angle - true_angle), np.cos(pred_angle - true_angle)))
                    ve = np.linalg.norm(pred_abs[4:6] - true_abs[4:6])  # vx, vy
                    ave = abs(pred_abs[6] - true_abs[6])

                    if h < horizon:
                        pos_err[h].append(pe)
                        angle_err[h].append(ae)
                        vel_err[h].append(ve)
                        ang_vel_err[h].append(ave)
                        pos_true_x[h].append(true_abs[0])
                        pos_pred_x[h].append(pred_abs[0])
                        pos_true_y[h].append(true_abs[1])
                        pos_pred_y[h].append(pred_abs[1])
                        angle_true[h].append(true_angle)
                        angle_pred[h].append(pred_angle)
                        heading_sin_true[h].append(true_abs[SIN_IDX])
                        heading_sin_pred[h].append(pred_abs[SIN_IDX])
                        heading_cos_true[h].append(true_abs[COS_IDX])
                        heading_cos_pred[h].append(pred_abs[COS_IDX])
                        vel_true_x[h].append(true_abs[4])
                        vel_pred_x[h].append(pred_abs[4])
                        vel_true_y[h].append(true_abs[5])
                        vel_pred_y[h].append(pred_abs[5])
                        ang_vel_true[h].append(true_abs[6])
                        ang_vel_pred[h].append(pred_abs[6])

                    # slide window: drop oldest, append predicted state +
                    # next real control input
                    next_control = controls[real_idx]
                    abs_state = np.vstack([abs_state[1:], pred_abs])
                    new_delta = pred_abs - cur_abs
                    delta_state = np.vstack([delta_state[1:], new_delta])
                    cur_controls = np.vstack([cur_controls[1:], next_control])

    horizons = np.arange(1, horizon + 1)
    pos_mean = [np.mean(e) if e else np.nan for e in pos_err]
    angle_mean = [np.mean(e) if e else np.nan for e in angle_err]
    vel_mean = [np.mean(e) if e else np.nan for e in vel_err]
    ang_vel_mean = [np.mean(e) if e else np.nan for e in ang_vel_err]
    pos_true_x_mean = [np.mean(e) if e else np.nan for e in pos_true_x]
    pos_pred_x_mean = [np.mean(e) if e else np.nan for e in pos_pred_x]
    pos_true_y_mean = [np.mean(e) if e else np.nan for e in pos_true_y]
    pos_pred_y_mean = [np.mean(e) if e else np.nan for e in pos_pred_y]
    vel_true_x_mean = [np.mean(e) if e else np.nan for e in vel_true_x]
    vel_pred_x_mean = [np.mean(e) if e else np.nan for e in vel_pred_x]
    vel_true_y_mean = [np.mean(e) if e else np.nan for e in vel_true_y]
    vel_pred_y_mean = [np.mean(e) if e else np.nan for e in vel_pred_y]
    ang_vel_true_mean = [np.mean(e) if e else np.nan for e in ang_vel_true]
    ang_vel_pred_mean = [np.mean(e) if e else np.nan for e in ang_vel_pred]

    def safe_r2(y_true, y_pred):
        if len(y_true) < 2:
            return np.nan
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if np.allclose(y_true, y_true[0]):
            return np.nan
        return r2_score(y_true, y_pred)

    r2_pos_x = [safe_r2(y_t, y_p) for y_t, y_p in zip(pos_true_x, pos_pred_x)]
    r2_pos_y = [safe_r2(y_t, y_p) for y_t, y_p in zip(pos_true_y, pos_pred_y)]
    r2_vel_x = [safe_r2(y_t, y_p) for y_t, y_p in zip(vel_true_x, vel_pred_x)]
    r2_vel_y = [safe_r2(y_t, y_p) for y_t, y_p in zip(vel_true_y, vel_pred_y)]
    r2_ang_vel = [safe_r2(y_t, y_p) for y_t, y_p in zip(ang_vel_true, ang_vel_pred)]
    r2_heading_sin = [safe_r2(y_t, y_p) for y_t, y_p in zip(heading_sin_true, heading_sin_pred)]
    r2_heading_cos = [safe_r2(y_t, y_p) for y_t, y_p in zip(heading_cos_true, heading_cos_pred)]
    r2_position = [np.nanmean(vals) for vals in zip(r2_pos_x, r2_pos_y)]
    r2_heading = [np.nanmean(vals) for vals in zip(r2_heading_sin, r2_heading_cos)]

    def circular_mean(values):
        if not values:
            return np.nan
        values = np.asarray(values, dtype=float)
        return np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values)))

    angle_true_mean = [circular_mean(e) for e in angle_true]
    angle_pred_mean = [circular_mean(e) for e in angle_pred]

    fig, axs = plt.subplots(4, 2, figsize=(14, 13), sharex='col')

    axs[0, 0].plot(horizons, pos_mean)
    axs[0, 0].set_title('Position error vs. rollout horizon (closed-loop)')
    axs[0, 0].set_ylabel('mean position error (m)')

    axs[0, 1].plot(horizons, pos_true_x_mean, label='true x')
    axs[0, 1].plot(horizons, pos_pred_x_mean, label='pred x')
    axs[0, 1].plot(horizons, pos_true_y_mean, label='true y')
    axs[0, 1].plot(horizons, pos_pred_y_mean, label='pred y')
    axs[0, 1].set_title('Position values vs. rollout horizon')
    axs[0, 1].set_ylabel('position (m)')
    axs[0, 1].legend(loc='best')

    axs[1, 0].plot(horizons, angle_mean)
    axs[1, 0].set_title('Heading error vs. rollout horizon')
    axs[1, 0].set_ylabel('mean |angle error| (rad)')

    axs[1, 1].plot(horizons, angle_true_mean, label='true angle')
    axs[1, 1].plot(horizons, angle_pred_mean, label='pred angle')
    axs[1, 1].set_title('Heading values vs. rollout horizon')
    axs[1, 1].set_ylabel('angle (rad)')
    axs[1, 1].legend(loc='best')

    axs[2, 0].plot(horizons, vel_mean)
    axs[2, 0].set_title('Linear velocity error vs. rollout horizon')
    axs[2, 0].set_xlabel('steps into rollout')
    axs[2, 0].set_ylabel('mean velocity error (m/s)')

    axs[2, 1].plot(horizons, vel_true_x_mean, label='true vx')
    axs[2, 1].plot(horizons, vel_pred_x_mean, label='pred vx')
    axs[2, 1].plot(horizons, vel_true_y_mean, label='true vy')
    axs[2, 1].plot(horizons, vel_pred_y_mean, label='pred vy')
    axs[2, 1].set_title('Linear velocity values vs. rollout horizon')
    axs[2, 1].set_xlabel('steps into rollout')
    axs[2, 1].set_ylabel('velocity (m/s)')
    axs[2, 1].legend(loc='best')

    axs[3, 0].plot(horizons, ang_vel_mean)
    axs[3, 0].set_title('Angular velocity error vs. rollout horizon')
    axs[3, 0].set_xlabel('steps into rollout')
    axs[3, 0].set_ylabel('mean angular velocity error (rad/s)')

    axs[3, 1].plot(horizons, ang_vel_true_mean, label='true angular_velocity_z')
    axs[3, 1].plot(horizons, ang_vel_pred_mean, label='pred angular_velocity_z')
    axs[3, 1].set_title('Angular velocity values vs. rollout horizon')
    axs[3, 1].set_xlabel('steps into rollout')
    axs[3, 1].set_ylabel('angular velocity (rad/s)')
    axs[3, 1].legend(loc='best')

    for row in range(4):
        axs[row, 0].set_xlabel('steps into rollout')

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

    if name.endswith('.png'):
        traj_name = name[:-4] + '_rollout_trajectories.png'
    else:
        traj_name = name + '_rollout_trajectories.png'
    _plot_trajectory_examples(
        model, list(segments), scaler_in, scaler_out, device,
        seq_length, horizon, traj_name, n_examples=6
    )

    print(f"Saved rollout evaluation plot to {name}")
    print(f"Saved rollout trajectory plot to {traj_name}")
    print(f"1-step pos err: {pos_mean[0]:.4f}  |  {horizon}-step pos err: {pos_mean[-1]:.4f}")
    print(f"R2 @ 1-step  pos={r2_position[0]:.4f}  heading={r2_heading[0]:.4f}  vx={r2_vel_x[0]:.4f}  vy={r2_vel_y[0]:.4f}  ang_vel={r2_ang_vel[0]:.4f}")
    print(f"R2 @ final   pos={r2_position[-1]:.4f}  heading={r2_heading[-1]:.4f}  vx={r2_vel_x[-1]:.4f}  vy={r2_vel_y[-1]:.4f}  ang_vel={r2_ang_vel[-1]:.4f}")


def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference (closed-loop rollout)")
    return input("Enter mode (1 or 2): ")


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = selectMode()

    if mode == '1':
        inputs, outputs = get_data(config['data_dir'], config['data_source'])

        # Explicitly map the names of the columns to guide the PhysicsScaler
        input_feature_names = CONTROL_COLS + [f"abs_{c}" for c in STATE_COLS if c not in POSITION_COLS] + [f"delta_{c}" for c in STATE_COLS]
        output_feature_names = [f"delta_{c}" for c in STATE_COLS]

        N, seq_len, num_features = np.shape(inputs)
        
        scaler_in = StandardScaler()
        inputs_flat = np.reshape(inputs, (N * seq_len, num_features))
        inputs_scaled = scaler_in.fit_transform(inputs_flat)
        inputs = np.reshape(inputs_scaled, (N, seq_len, num_features))
        joblib.dump(scaler_in, 'scaler_in.pkl')

        scaler_out = StandardScaler()
        outputs = scaler_out.fit_transform(outputs)
        joblib.dump(scaler_out, 'scaler_out.pkl')

        X_train, X_val, y_train, y_val = train_test_split(
            inputs, outputs, test_size=config['validation_split'], random_state=42)

        train_dataset = TensorDataset(torch.tensor(np.array(X_train), dtype=torch.float32),
                                       torch.tensor(np.array(y_train), dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        val_dataset = TensorDataset(torch.tensor(np.array(X_val), dtype=torch.float32),
                                     torch.tensor(np.array(y_val), dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        model = CarDynamicModel(input_size=num_features, output_size=np.shape(outputs)[1])
        model.to(device)
        log_dir = os.path.join(os.path.dirname(__file__), 'logs', config['log_dir'])
        writer = SummaryWriter(log_dir=log_dir)

        # NOTE: checkpoint_dir is now separate from the final model save
        # path -- previously both used config['model_save_path'], which
        # would crash (os.makedirs on a path that a later torch.save also
        # tries to write to as a file).
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')

        # Segments for closed-loop multi-step training (same continuous,
        # no-reset chunks used by rollout evaluation, just from the
        # training files instead of the test file).
        all_multistep_segments = []
        for f in config['data_source']:
            all_multistep_segments.extend(load_trajectory_segments(
                config['data_dir'], f, seq_length=config.get('multistep_seq_length', SEQ_LENGTH)))

        multistep_segments = all_multistep_segments
        rollout_val_segments = None
        rollout_validation_split = config.get('rollout_validation_split', 0.2)
        if all_multistep_segments and rollout_validation_split > 0:
            rng = np.random.default_rng(config.get('rollout_validation_seed', 42))
            segment_indices = rng.permutation(len(all_multistep_segments))
            if len(segment_indices) > 1:
                split_idx = int(len(segment_indices) * (1.0 - rollout_validation_split))
                split_idx = max(1, min(split_idx, len(segment_indices) - 1))
                train_segment_indices = segment_indices[:split_idx]
                rollout_segment_indices = segment_indices[split_idx:]
                multistep_segments = [all_multistep_segments[i] for i in train_segment_indices]
                rollout_val_segments = [all_multistep_segments[i] for i in rollout_segment_indices]
            else:
                rollout_val_segments = list(all_multistep_segments)

        model = train_model(
            model, train_loader, val_loader, config['epochs'], config['learning_rate'],
            writer, device, checkpoint_dir=checkpoint_dir, patience=config.get('patience', 15),
            noise_std_max=config.get('noise_std_max', 0.02),
            noise_warmup_epochs=config.get('noise_warmup_epochs', 10),
            multistep_segments=multistep_segments if config.get('use_multistep_training', True) else None,
            scaler_in=scaler_in, scaler_out=scaler_out,
            rollout_val_segments=rollout_val_segments if config.get('use_multistep_training', True) else None,
            rollout_val_horizon=config.get('rollout_validation_horizon', config.get('multistep_k', 10)),
            rollout_val_max_segments=config.get('rollout_validation_max_segments', 6),
            rollout_val_starts_per_segment=config.get('rollout_validation_starts_per_segment', 3),
            multistep_k=config.get('multistep_k', 10),
            multistep_batches_per_epoch=config.get('multistep_batches_per_epoch', 20),
            multistep_batch_size=config.get('multistep_batch_size', 32),
            multistep_seq_length=config.get('multistep_seq_length', SEQ_LENGTH),
        )

        torch.save(model.state_dict(), config['model_save_path'])
        writer.close()

    elif mode == '2':
        scaler_in = joblib.load('scaler_in.pkl')
        scaler_out = joblib.load('scaler_out.pkl')

        model = CarDynamicModel(input_size=N_CONTROL + N_ABS_STATE + N_STATE, output_size=N_STATE)
        model.load_state_dict(torch.load(config['model_save_path'], map_location=device))
        model.to(device)

        evaluate_model_rollout(model, config['data_dir'], config['test_data_source'], device,
                                scaler_in, scaler_out, seq_length=SEQ_LENGTH,
                                horizon=config.get('rollout_horizon', 50),
                                name=config['result_png'])
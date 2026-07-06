import os
import json
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import get_data, WeightedMSELoss, get_test_data
from motorDynamicModel import MotorDynamicModel
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, val_loader, epochs, learning_rate, writer, device, name='model.pth'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error is perfect for this
    # weights = torch.tensor([10.0, 1.0, 0.1]).to(device) 
    # criterion = WeightedMSELoss(weights)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
        
            # Forward pass
            predictions = model(inputs)
            
            # Calculate loss (comparing predicted deltas to actual deltas)
            loss = criterion(predictions, targets)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_predictions = model(val_inputs)
                val_loss += criterion(val_predictions, val_targets).item()

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Validation', val_loss / len(val_loader), epoch)
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

        if (epoch + 1) % 10 == 0:  # Save model every 10 epochs
            if not os.path.exists(name):
                os.makedirs(name)
            torch.save(model.state_dict(), f'{name}/epoch_{epoch+1}.pth')


def evaluate_model(model, test_loader, device, scaler_in, scaler_out, name='test_results'):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    # 建立空列表來收集「所有」的預測值與真實值
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            
            # 計算 Loss
            test_loss += criterion(predictions, targets).item()
            
            # 將每個 batch 的結果轉回 CPU 並收集起來
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # 將 List 中的多個 Batch 拼接成一個完整的 Numpy Array
    # 維度會變成 (總樣本數, 輸出維度)
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"Test Loss: {test_loss / len(test_loader)}")

    predicted_deltas = scaler_out.inverse_transform(all_predictions)
    actual_deltas = scaler_out.inverse_transform(all_targets)

    last_step_inputs = all_inputs[:, -1, :] 
    last_step_inputs_real = scaler_in.inverse_transform(last_step_inputs)

    base_sin = last_step_inputs_real[:, 0]
    base_cos = last_step_inputs_real[:, 1]
    base_angle = np.arctan2(base_sin, base_cos) 
    base_velocity = last_step_inputs_real[:, 2]
    target_effort = last_step_inputs_real[:, 3]

    pred_abs_angle = base_angle + predicted_deltas[:, 0]
    act_abs_angle = base_angle + actual_deltas[:, 0]
    
    pred_abs_vel = base_velocity + predicted_deltas[:, 1]
    act_abs_vel = base_velocity + actual_deltas[:, 1]

    
    # 根據維度數量動態建立子圖
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows, 1 column

    # 設定 X 軸的時間步長
    time_steps = range(len(all_predictions))[:100]  # 只繪製前 1000 個時間步長以免圖表過於擁擠

    # 圖表 1: Target Effort
    axs[0].plot(time_steps, target_effort[:100], label='Target Effort', color='orange', alpha=0.8)
    axs[0].set_title('Input Target Effort (Real Values)')
    axs[0].legend()

    # 圖表 2: Angle (弧度)
    axs[1].plot(time_steps, pred_abs_angle[:100], label='Predicted Angle', color='blue', alpha=0.8)
    axs[1].plot(time_steps, act_abs_angle[:100], label='Actual Angle', color='green', alpha=0.8)
    axs[1].plot(time_steps, (pred_abs_angle - act_abs_angle)[:100], label='Error', color='red', alpha=0.8)
    axs[1].set_title('Absolute Angle (Radians)')
    axs[1].legend()
    print(f'Angle: Mean Absolute Error = {np.mean(np.abs(pred_abs_angle - act_abs_angle))}, Mean Squared Error = {np.mean((pred_abs_angle - act_abs_angle)**2)}')

    # 圖表 3: Velocity
    axs[2].plot(time_steps, pred_abs_vel[:100], label='Predicted Velocity', color='blue', alpha=0.8)
    axs[2].plot(time_steps, act_abs_vel[:100], label='Actual Velocity', color='green', alpha=0.8)
    axs[2].plot(time_steps, (pred_abs_vel - act_abs_vel)[:100], label='Error', color='red', alpha=0.8)
    axs[2].set_title('Absolute Velocity')
    axs[2].legend()
    print(f'Velocity: Mean Absolute Error = {np.mean(np.abs(pred_abs_vel - act_abs_vel))}, Mean Squared Error = {np.mean((pred_abs_vel - act_abs_vel)**2)}')

    plt.tight_layout()
    plt.savefig(f'{name}')
    plt.close()

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode


if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Get data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = selectMode()
    if mode == '1':
        inputs, outputs = get_data(config['data_dir'], config['data_source'])

        N, seq_len, num_features = np.shape(inputs)
        scaler_in = StandardScaler()
        inputs_flat = np.reshape(inputs, (N * seq_len, num_features))
        inputs_scaled = scaler_in.fit_transform(inputs_flat)
        inputs = np.reshape(inputs_scaled, (N, seq_len, num_features))
        joblib.dump(scaler_in, 'scaler_in.pkl')  # Save the input scaler for later use

        # Scale outputs
        scaler_out = StandardScaler()
        outputs = scaler_out.fit_transform(outputs)
        joblib.dump(scaler_out, 'scaler_out.pkl')  # Save the output scaler for later use
        
        # Split to training and testing sets, then further split training to training and validation
        X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=config['validation_split'], random_state=42)

        # Create DataLoader for training
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        # Create DataLoader for validation
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Train the model
        model = MotorDynamicModel()
        model.to(device)
        log_dir = os.path.join(os.path.dirname(__file__), 'logs', config['log_dir'])
        writer = SummaryWriter(log_dir=log_dir)
        train_model(model, train_loader, val_loader, config['epochs'], config['learning_rate'], writer, device, name=config['model_save_path'])
        # Save the model
        torch.save(model.state_dict(), config['model_save_path'])
        writer.close()
    elif mode == '2':
        # Load test data
        test_inputs, test_outputs = get_test_data(config['data_dir'], config['test_data_source'])

        N_test, seq_len_test, num_features_test = np.shape(test_inputs)
        scaler_in = joblib.load('scaler_in.pkl')  # Load the input scaler
        test_inputs_flat = np.reshape(test_inputs, (N_test * seq_len_test, num_features_test))
        test_inputs_scaled = scaler_in.transform(test_inputs_flat)
        test_inputs = np.reshape(test_inputs_scaled, (N_test, seq_len_test, num_features_test))

        scaler_out = joblib.load('scaler_out.pkl')  # Load the output scaler
        test_outputs = scaler_out.transform(test_outputs)

        # Create DataLoader for testing
        test_inputs = np.array(test_inputs)
        test_outputs = np.array(test_outputs)
        test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.float32), torch.tensor(test_outputs, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        # Load the model
        model = MotorDynamicModel()
        model.load_state_dict(torch.load(config['model_save_path'], map_location=device))
        model.to(device)
        # Evaluate the model on the test set
        evaluate_model(model, test_loader, device, scaler_in, scaler_out, name=config['result_png'])
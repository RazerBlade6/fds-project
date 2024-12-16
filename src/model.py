import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path, stock_name):
    # with pd.HDFStore(file_path) as store:
    #     data = store[stock_name]  # Load data for the given stock
    #     print(data)
    # data['timestamp'] = pd.to_datetime(data['timestamp'])
    # data.sort_values('timestamp', inplace=True)
    # return data[['close']].values
    print(file_path)

    with open(file_path,encoding='UTF-8') as f:
        print(f)
        data = pd.read_csv(f)
    print(data)
    return data[['close']].values

def prepare_data(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_and_plot(model, X_test, y_test, scaler, stock_name):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    
    # Reverse scaling
    y_test = scaler.inverse_transform(y_test.numpy())
    y_pred = scaler.inverse_transform(y_pred)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'Stock Price Prediction for {stock_name}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def process_stock(file_path, stock_name, sequence_length=60, batch_size=32, hidden_size=50, 
                  num_layers=2, num_epochs=50, learning_rate=0.001):
    data = load_data(file_path, stock_name)
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    X, y = prepare_data(data, sequence_length)
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    
    predict_and_plot(model, X_test.to(device), y_test, scaler, stock_name)

def main():
    file_path = "./data/FullDataCsv/AXISBANK__EQ__NSE__NSE__MINUTE.csv"
    stock_names = ["AXISBANK__EQ__NSE__NSE__MINUTE"]
    for stock_name in stock_names:
        print(f"\nProcessing {stock_name}...")
        process_stock(file_path, stock_name)

if __name__ == "__main__":
    main()

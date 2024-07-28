import flwr as fl
from typing import Dict, List, Tuple
from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys

# Cấu hình client
CLIENT_ID = int(sys.argv[1])  # Lấy id của client từ command line argument
NUM_CLASSES = 8  # Số lượng nhãn dịch vụ
EPOCHS = 10
BATCH_SIZE = 32


# Đọc dữ liệu từ file feather
def load_data():
  return pd.read_feather(f"data/df_train_part_{CLIENT_ID}.feather")


# Chuẩn hóa nhãn dịch vụ
def normalize_labels(labels):
  label_map = {1: 0, 4: 1, 5: 2, 8: 3, 9: 4, 11: 5, 14: 6, 16: 7}
  return np.array([label_map[label] for label in labels])

# Chuẩn hóa dữ liệu và reshape
def preprocess_data(df):
  # Gom nhóm các packet theo flow_id
  flows = df.groupby('flow_id')
  
  X = []
  y = []
  for flow_id, flow in flows:
    # Lấy dữ liệu của flow
    flow_data = flow.iloc[:, :128].values
    if flow_data.shape[0] > 20:
        flow_data = flow_data[:20]
    # Padding nếu số lượng packet nhỏ hơn 20
    if flow_data.shape[0] < 20:
      padding_shape = (20 - flow_data.shape[0], 128)
      flow_data = np.concatenate((flow_data, np.zeros(padding_shape)))

    X.append(flow_data)
    y.append(flow['Label'].iloc[0])
    
  X = np.array(X)
  y = normalize_labels(y)
  return X, y

# Tạo model giống model trên server
def create_model(model_architecture="simple"):
    """Tạo model với lựa chọn kiến trúc."""
    if model_architecture == "simple":
        model = keras.models.Sequential([
            keras.layers.Input(shape=(20, 128)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
    elif model_architecture == "cnn":
        model = keras.models.Sequential([
            keras.layers.Input(shape=(20, 128)),
            keras.layers.Reshape((20, 128, 1)),  # Reshape cho CNN
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
    else:
        raise ValueError(f"Kiến trúc '{model_architecture}' không hợp lệ.")

    return model

# Khởi tạo client
class FlowerClient(fl.client.NumPyClient):
  def __init__(self, client_id, model_architecture="simple"):
    self.client_id = client_id
    self.model = create_model(model_architecture)  # Truyền kiến trúc mong muốn
    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    self.x_train, self.y_train = self.load_data()

  def get_parameters(self, config):
    return self.model.get_weights()

  def fit(self, parameters, config):
    self.model.set_weights(parameters)
    self.model.fit(self.x_train, self.y_train, epochs=config["epochs"], batch_size=config["batch_size"])
    return self.model.get_weights(), len(self.x_train), {}

  def evaluate(self, parameters, config):
    self.model.set_weights(parameters)
    df_test = pd.read_feather("data/df_test.feather")
    x_test, y_test = preprocess_data(df_test)
    loss, accuracy = self.model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}

  def load_data(self):
    df = load_data()
    x_train, y_train = preprocess_data(df)
    return x_train, y_train

# Khởi chạy client với lựa chọn kiến trúc
model_architecture = "cnn"  # Chọn kiến trúc ("simple" hoặc "cnn")
fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(CLIENT_ID, model_architecture),  # Truyền kiến trúc cho client
)
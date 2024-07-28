import flwr as fl
from typing import Dict
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Cấu hình
NUM_CLIENTS = 7  # Số lượng client (điền giá trị thực tế)
EPOCHS = 10  # Số lượng epoch
BATCH_SIZE = 32
NUM_CLASSES = 8
LEARNING_RATE = 0.001

# Tạo mô hình với lựa chọn kiến trúc
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
            keras.layers.Reshape((20, 128, 1)), # Reshape cho CNN
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
    else:
        raise ValueError(f"Kiến trúc '{model_architecture}' không hợp lệ.")

    return model

# Khởi tạo model trên server
model = create_model(model_architecture="cnn") # Chọn kiến trúc ("simple" hoặc "cnn")
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Định nghĩa strategy cho Federated Learning
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
)


def get_on_fit_begin_fn(tensorboard_callback):
    """Hàm trả về on_fit_begin_fn."""
    def on_fit_begin(server_round):
        print(f"Starting round {server_round}!")
        # Tạo TensorBoard callback mới cho mỗi vòng huấn luyện
        log_dir = f"./logs/round_{server_round}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        return {"callback": [tensorboard_callback]} # Trả về callback trong dict

    return on_fit_begin


# Khởi tạo TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

# Gán hàm on_fit_begin_fn cho strategy sau khi khởi tạo
strategy.on_fit_begin = get_on_fit_begin_fn(tensorboard_callback)

# Bắt đầu server với TensorBoard callback
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
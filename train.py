import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Create directories for saving models and visualizations
os.makedirs('trained_models', exist_ok=True)
os.makedirs('performance_metrics', exist_ok=True)

# Load and preprocess the data
data = pd.read_csv('/Users/arpantiwari/Downloads/Projects/MiniProject/code/processed_data.csv')

# Assuming the last column is the label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape X for CNN input
X = X[..., np.newaxis]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define CNN-BiLSTM model
def build_cnn_bilstm(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define Transformer model
def build_transformer(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Permute((2, 1))(inputs)  # Transformer expects channel-last
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot training and validation metrics
def plot_metrics(history, model_name):
    plt.figure(figsize=(12, 6))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'performance_metrics/{model_name}_metrics.png')
    plt.close()

# Callbacks
checkpoint_cb_cnn_bilstm = ModelCheckpoint('trained_models/cnn_bilstm_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
checkpoint_cb_transformer = ModelCheckpoint('trained_models/transformer_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping_cb = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)



# Train CNN-BiLSTM
cnn_bilstm_model = build_cnn_bilstm(input_shape=(X_train.shape[1], 1), num_classes=y.shape[1])
history_cnn_bilstm = cnn_bilstm_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint_cb_cnn_bilstm, early_stopping_cb]
)
plot_metrics(history_cnn_bilstm, 'CNN-BiLSTM')

# Train Transformer
transformer_model = build_transformer(input_shape=(X_train.shape[1], 1), num_classes=y.shape[1])
history_transformer = transformer_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint_cb_transformer, early_stopping_cb]
)
plot_metrics(history_transformer, 'Transformer')

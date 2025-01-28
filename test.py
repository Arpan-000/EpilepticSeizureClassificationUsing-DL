import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory for saving test performance results
os.makedirs('test_performance', exist_ok=True)

# Load and preprocess the data
data = pd.read_csv('processed_data.csv')

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

# Load models
cnn_bilstm_model = load_model('trained_models/cnn_bilstm_model.h5')
transformer_model = load_model('trained_models/transformer_model.h5')

# Evaluate CNN-BiLSTM model
cnn_bilstm_eval = cnn_bilstm_model.evaluate(X_test, y_test, verbose=0)
cnn_bilstm_preds = cnn_bilstm_model.predict(X_test)
cnn_bilstm_preds_classes = np.argmax(cnn_bilstm_preds, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

cnn_bilstm_report = classification_report(y_test_classes, cnn_bilstm_preds_classes, target_names=label_encoder.classes_, output_dict=True)
cnn_bilstm_cm = confusion_matrix(y_test_classes, cnn_bilstm_preds_classes)

with open('test_performance/cnn_bilstm_evaluation.txt', 'w') as f:
    f.write(f"Loss: {cnn_bilstm_eval[0]:.4f}\nAccuracy: {cnn_bilstm_eval[1]:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(pd.DataFrame(cnn_bilstm_report).transpose().to_string())
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(cnn_bilstm_cm))

plt.figure(figsize=(10, 7))
sns.heatmap(cnn_bilstm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('CNN-BiLSTM Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('test_performance/cnn_bilstm_confusion_matrix.png')
plt.close()

# Evaluate Transformer model
transformer_eval = transformer_model.evaluate(X_test, y_test, verbose=0)
transformer_preds = transformer_model.predict(X_test)
transformer_preds_classes = np.argmax(transformer_preds, axis=1)

transformer_report = classification_report(y_test_classes, transformer_preds_classes, target_names=label_encoder.classes_, output_dict=True)
transformer_cm = confusion_matrix(y_test_classes, transformer_preds_classes)

with open('test_performance/transformer_evaluation.txt', 'w') as f:
    f.write(f"Loss: {transformer_eval[0]:.4f}\nAccuracy: {transformer_eval[1]:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(pd.DataFrame(transformer_report).transpose().to_string())
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(transformer_cm))

plt.figure(figsize=(10, 7))
sns.heatmap(transformer_cm, annot=True, fmt='d', cmap='Greens', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Transformer Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('test_performance/transformer_confusion_matrix.png')
plt.close()

# Plot training and validation metrics (example for accuracy and loss, if available)
# Assuming the models were trained with accuracy and loss metrics tracked
for model_name, model_path in zip(['cnn_bilstm', 'transformer'], ['trained_models/cnn_bilstm_model.h5', 'trained_models/transformer_model.h5']):
    model = load_model(model_path)
    history = model.history if hasattr(model, 'history') else None

    if history:
        # Accuracy Plot
        plt.figure()
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name.upper()} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'test_performance/{model_name}_accuracy.png')
        plt.close()

        # Loss Plot
        plt.figure()
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name.upper()} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'test_performance/{model_name}_loss.png')
        plt.close()

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def load_data(data_dir):
    data = []
    labels = []
    
    # Read labels from CSV
    labels_df = pd.read_csv('dataset/labels.csv', skiprows=2)
    
    # Define category directories
    categories = ['numerals', 'vowels', 'consonants']
    
    # Process each category
    for category in categories:
        category_dir = os.path.join(data_dir, 'nhcd', 'nhcd', category)
        if os.path.exists(category_dir):
            # Process each class directory within the category
            for class_dir in os.listdir(category_dir):
                class_path = os.path.join(category_dir, class_dir)
                if os.path.isdir(class_path):
                    try:
                        class_idx = int(class_dir)  # Convert directory name to class index
                        # Process all images in this class directory
                        for img_name in os.listdir(class_path):
                            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(class_path, img_name)
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    img = cv2.resize(img, (32, 32))
                                    data.append(img)
                                    labels.append(class_idx)
                    except ValueError:
                        print(f"Warning: Could not convert directory name to class index: {class_dir}")
                        continue
    
    if len(data) == 0:
        raise ValueError("No images were loaded. Please check the dataset directory structure and image files.")
    
    print(f"Loaded {len(data)} images with {len(np.unique(labels))} unique classes")
    return np.array(data), np.array(labels)

def create_model(num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def save_model_and_metadata(model, num_classes, class_mapping):
    """
    Save the trained model and its metadata
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    model_path = 'models/devanagari_model.h5'
    model.save(model_path)
    
    # Save metadata
    metadata = {
        'num_classes': num_classes,
        'class_mapping': class_mapping,
        'input_shape': (32, 32, 1)
    }
    np.save('models/model_metadata.npy', metadata)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to models/model_metadata.npy")

def load_saved_model():
    """
    Load the saved model and its metadata
    """
    model_path = 'models/devanagari_model.h5'
    metadata_path = 'models/model_metadata.npy'
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Model or metadata files not found. Please train the model first.")
    
    # Load the model
    model = load_model(model_path)
    
    # Load metadata
    metadata = np.load(metadata_path, allow_pickle=True).item()
    
    return model, metadata

def train_model():
    # Load and preprocess data
    data_dir = 'dataset'
    X, y = load_data(data_dir)
    
    # Get number of unique classes
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    
    # Create class mapping
    class_mapping = {i: str(i) for i in range(num_classes)}
    
    # Reshape and normalize data
    X = X.reshape(-1, 32, 32, 1) / 255.0
    y = to_categorical(y, num_classes=num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    model = create_model(num_classes=num_classes)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Print model summary
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint('best_model.h5',
                               monitor='val_accuracy',
                               save_best_only=True,
                               mode='max',
                               verbose=1)
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                 patience=10,
                                 restore_best_weights=True)
    
    # Train model
    history = model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=50,
                       validation_data=(X_test, y_test),
                       callbacks=[checkpoint, early_stopping])
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save the trained model and metadata
    save_model_and_metadata(model, num_classes, class_mapping)
    
    return model

if __name__ == "__main__":
    model = train_model() 
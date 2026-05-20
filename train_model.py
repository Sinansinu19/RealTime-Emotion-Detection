import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, RandomFlip, RandomRotation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Reset Keras background memory
tf.keras.backend.clear_session()

# 1. Dataset path
base_dir = 'dataset'

# 2. Modern Data Loading 
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale"
)

# Print the found class names in terminal so you know the exact ordering
class_names = train_ds.class_names
print("Detected classes in folder structure:", class_names)
num_classes = len(class_names)

# 3. CNN Architecture
model = Sequential([
    Input(shape=(48, 48, 1)),         
    Rescaling(1./255),               
    RandomFlip("horizontal"),        
    RandomRotation(0.1),             
    
    # Conv Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Conv Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Conv Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    # Classification Head
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), 
    Dense(num_classes, activation='softmax') # Dynamic fix: automatically scales to 5 classes
], name="emotion_detection_network")

# 4. Compile
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks 
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'emotion_model.keras',        
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max',
    verbose=1
)

# 6. Run Training
print("Starting training script...")
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[early_stop, checkpoint]
)

print("✅ Success! Model trained and saved cleanly to emotion_model.keras")
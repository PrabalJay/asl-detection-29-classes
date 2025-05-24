import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore') 

import tensorflow as tf
from tensorflow import keras

# Configuration
config = {
    'img_size': 100,
    'batch_size': 128,
    'epochs': 20, 
    'num_classes': 29,
    'train_path': os.path.join('data', 'train'),
    'model_path': 'models/model.keras'
}

# Model Directory
os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)

# Data Pipeline
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10, 
    width_shift_range=0.1,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    config['train_path'],
    target_size=(config['img_size'], config['img_size']),
    batch_size=config['batch_size'],
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    config['train_path'],
    target_size=(config['img_size'], config['img_size']),
    batch_size=config['batch_size'],
    class_mode='categorical',
    subset='validation'
)

# Optimized Model
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(config['img_size'], config['img_size'], 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(config['num_classes'], activation='softmax')
    ])
    return model

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
print(f"\nTraining on {train_gen.samples:,} images")
print(f"Validating on {val_gen.samples:,} images\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=config['epochs'],
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            config['model_path'],
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ],
    verbose=1
)

print(f"\nTraining complete. Model saved to {config['model_path']}")
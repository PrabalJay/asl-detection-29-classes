import os

class Config:
    # Image parameters
    IMG_WIDTH = 100
    IMG_HEIGHT = 100
    CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 29 
    
    # Paths
    TRAIN_DATA_PATH = os.path.join('data', 'train')
    TEST_DATA_PATH = os.path.join('data', 'test')
    MODEL_SAVE_PATH = 'models'
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
    
    # Class names
    CLASS_NAMES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]

# Create directories
os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
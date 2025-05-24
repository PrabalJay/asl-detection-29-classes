import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

EXPECTED_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

def load_data():
    test_dir = os.path.join('data', 'test')
    datagen = ImageDataGenerator(rescale=1./255)
    
    data_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(100, 100),
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    
    class_mapping = {cls: idx for idx, cls in enumerate(EXPECTED_CLASSES)}
    actual_classes = sorted(data_gen.class_indices.keys())
    
    print(f"Found {data_gen.samples} images in {len(actual_classes)} classes: {actual_classes}")
    
    missing = set(EXPECTED_CLASSES) - set(actual_classes)
    if missing:
        print(f"Warning: Missing test images for classes: {missing}")
    
    return data_gen, class_mapping

def main():
    try:
        model = keras.models.load_model('models/model.keras')
        data_gen, class_mapping = load_data()
        
        # Evaluate
        results = model.evaluate(data_gen, verbose=0)
        print(f"\nEvaluation Results:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]*100:.2f}%")
        
        # Predictions
        y_pred = model.predict(data_gen, verbose=0)
        y_true = data_gen.classes
        
        y_pred_full = np.zeros((len(y_true), len(EXPECTED_CLASSES)))
        for i, pred in enumerate(y_pred):
            y_pred_full[i, :len(pred)] = pred
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true,
            np.argmax(y_pred, axis=1),
            labels=[class_mapping[cls] for cls in data_gen.class_indices.keys()],
            target_names=EXPECTED_CLASSES,
            zero_division=0
        ))
        
        # Confusion matrix
        plt.figure(figsize=(20, 18))
        cm = tf.math.confusion_matrix(y_true, np.argmax(y_pred, axis=1))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=sorted(data_gen.class_indices.keys()),
            yticklabels=sorted(data_gen.class_indices.keys())
        )
        plt.title('Confusion Matrix (Present Classes Only)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("\nSaved confusion_matrix.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure each class in data/test/ has at least one image")
        print("2. Check folder names exactly match: A-Z + del + nothing + space")
        print("3. Verify model expects 29 output classes")

if __name__ == "__main__":
    main()
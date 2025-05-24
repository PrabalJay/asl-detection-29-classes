from PIL import Image
import os

test_dir = os.path.join('data', 'test')
for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    for img_file in os.listdir(class_dir):
        try:
            img_path = os.path.join(class_dir, img_file)
            with Image.open(img_path) as img:
                img.verify()  # Verify image integrity
            print(f"✓ {img_path} is valid")
        except Exception as e:
            print(f"✗ {img_path} is corrupted: {str(e)}")
            os.remove(img_path)  # Delete corrupted file
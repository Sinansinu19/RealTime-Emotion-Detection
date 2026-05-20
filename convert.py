import os
from PIL import Image

# Your dataset folder
dataset_dir = 'dataset'
emotions = ['HAPPY', 'SAD'] # Matching your exact folder names in uppercase

count = 0

for emotion in emotions:
    folder_path = os.path.join(dataset_dir, emotion)
    
    if not os.path.exists(folder_path):
        print(f"Could not find folder: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".webp"):
            webp_path = os.path.join(folder_path, filename)
            jpg_path = os.path.join(folder_path, filename.replace(".webp", ".jpg"))
            
            try:
                # Open webp, convert to standard RGB, and save as jpg
                img = Image.open(webp_path).convert("RGB")
                img.save(jpg_path, "JPEG")
                
                # Delete the original webp file so Keras doesn't get confused
                os.remove(webp_path)
                count += 1
            except Exception as e:
                print(f"Error converting {filename}: {e}")

print(f"✅ Success! Converted {count} images to .jpg format.")
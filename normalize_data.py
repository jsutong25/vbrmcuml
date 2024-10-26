import os
from PIL import Image

# Folder where your label files are stored
label_folder = './yolov5/dataset/mangoes/labels/train'
image_folder = './yolov5/dataset/mangoes/images/train'

# Iterate through the label files
for label_file in os.listdir(label_folder):
    if label_file.endswith('.txt') and label_file.startswith('freshMango'):
        label_path = os.path.join(label_folder, label_file)
        # Find corresponding image to get dimensions
        image_path = os.path.join(image_folder, label_file.replace('.txt', '.jpg'))
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found.")
            continue
        
        # Open image to get its dimensions
        with open(image_path, 'rb') as img_file:
            img = Image.open(img_file)
            img_width, img_height = img.size
        
        # Read the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            values = line.strip().split()
            class_id = values[0]
            # Assuming the label format is [class_id, x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(float, values[1:])
            
            # Ensure that the bounding box coordinates are non-negative and within the image dimensions
            x_min = max(0, min(x_min, img_width))
            y_min = max(0, min(y_min, img_height))
            x_max = max(0, min(x_max, img_width))
            y_max = max(0, min(y_max, img_height))
            
            # Normalize the coordinates
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            box_width = (x_max - x_min) / img_width
            box_height = (y_max - y_min) / img_height

            # Skip boxes that are invalid or have zero width/height
            if box_width <= 0 or box_height <= 0:
                print(f"Warning: Invalid box in {label_file}. Skipping line.")
                continue
            
            # Append new normalized values
            new_lines.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
        
        # Write new normalized label file if valid lines exist
        if new_lines:
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
            print(f"Normalized {label_file}")
        else:
            print(f"Warning: No valid bounding boxes in {label_file}, skipping file.")

import os

label_folder = './yolov5/valid/labels'  # Adjust folder path as needed

for label_file in os.listdir(label_folder):
    if label_file.endswith('.txt'):
        file_path = os.path.join(label_folder, label_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Replace '0' (or whatever the old Mango class ID was) with '10'
        new_lines = [line.replace('0', '10') for line in lines]
        
        with open(file_path, 'w') as file:
            file.writelines(new_lines)
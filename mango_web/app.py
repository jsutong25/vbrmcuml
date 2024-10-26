from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from werkzeug.utils import secure_filename
from shutil import copyfile
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the YOLOv5 models
mango_model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp24/weights/best.pt', source='local')
ripeness_model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp6/weights/best.pt', source='local')
disease_model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp18/weights/best.pt', source='local')  # Load disease model

@app.route('/', methods=['GET', 'POST'])
def index():
    mango_message = ""  # Variable to store mango detection result
    ripeness_label = ""  # Variable to store ripeness classification result
    disease_label = ""  # Variable to store disease detection result
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run YOLOv5 mango detection inference with the merged dataset model
            mango_results = mango_model(filepath)
            
            # Get the names of the detected classes (including mango and non-mango objects)
            detected_classes = mango_results.names  # class labels
            
            # Check if 'mango' or similar class exists in the detected objects
            mango_detected = False
            for detection in mango_results.pred[0]:  # Iterate over predictions
                class_id = int(detection[-1])  # Class ID from detection
                if detected_classes[class_id].lower() == 'mango':  # Adjust 'mango' or 'Mango'
                    mango_detected = True
                    break

            # Set the mango message based on detection
            mango_message = "It's a mango!" if mango_detected else "Not a mango."

            # Save results (with bounding boxes) in the same folder as the uploaded image
            mango_results.save()
            
            time.sleep(2)

            # Get the correct "exp" folder name by filtering out non-standard folder names
            detect_path = 'runs/detect'
            exp_folders = [folder for folder in os.listdir(detect_path) if folder.startswith('exp') and folder[3:].isdigit()]

            if exp_folders:
                latest_exp_folder = sorted(exp_folders, key=lambda x: int(x.replace('exp', '')))[-1]
                detect_folder = os.path.join(detect_path, latest_exp_folder)

                # Attempt to find the mango result image, first by checking the .jpg extension
                result_img_path = os.path.join(detect_folder, os.path.splitext(filename)[0] + '.jpg')
                
                # If the result is not saved as .jpg, fallback to the original extension
                if not os.path.exists(result_img_path):
                    result_img_path = os.path.join(detect_folder, filename)
                    
                # Ensure the file actually exists before copying
                if os.path.exists(result_img_path):
                    # Move the result image to 'static/uploads' for easy serving
                    destination_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(result_img_path))
                    copyfile(result_img_path, destination_path)
                else:
                    return f"Error: The result image '{result_img_path}' does not exist."
            else:
                return "Error: Could not find any 'exp' folder in 'runs/detect'."

            # Ripeness and disease detection if mango is detected
            if mango_message == "It's a mango!":
                # Ripeness detection
                ripeness_results = ripeness_model(filepath)
                ripeness_results.save()
                time.sleep(2)

                # Disease detection
                disease_results = disease_model(filepath)
                disease_results.save()
                time.sleep(2)

                # Get ripeness result
                if len(ripeness_results.pred) > 0 and len(ripeness_results.pred[0]) > 0:
                    ripeness_class_id = int(ripeness_results.pred[0][0][-1])
                    ripeness_label = ['over-ripe', 'partially-ripe', 'ripe', 'unripe'][ripeness_class_id]
                else:
                    ripeness_label = "No ripeness detected."

                # Get disease result
                if len(disease_results.pred) > 0 and len(disease_results.pred[0]) > 0:
                    disease_class_id = int(disease_results.pred[0][0][-1])
                    disease_label = disease_results.names[disease_class_id]  # Get the detected disease name
                else:
                    disease_label = "No disease detected."

                return render_template('index.html', 
                                        uploaded_image=filepath, 
                                        mango_result_image=destination_path, 
                                        ripeness_label=ripeness_label,
                                        disease_label=disease_label,
                                        mango_message=mango_message)
            else:
                return render_template('index.html', 
                                       uploaded_image=filepath, 
                                       mango_result_image=destination_path, 
                                       mango_message=mango_message)

    return render_template('index.html', mango_message=mango_message, ripeness_label=ripeness_label, disease_label=disease_label)

if __name__ == '__main__':
    app.run(debug=True)

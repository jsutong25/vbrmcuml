from flask import Flask, render_template, request, redirect, url_for, Response
import os
import torch
from werkzeug.utils import secure_filename
from shutil import copyfile
import time
from data import disease
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the YOLOv5 models
mango_model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp24/weights/best.pt', source='local')
ripeness_model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp35/weights/best.pt', source='local')
disease_model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp18/weights/best.pt', source='local')  # Load disease model

mango_message = ""
ripeness_label = ""
disease_label = ""
camera_active = False

def process_frame(frame):
    global mango_message, ripeness_label, disease_label

    # Convert frame to the format YOLO expects without saving to file
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run mango model detection
    mango_results = mango_model(img_rgb)
    mango_detected = False

    # Iterate over mango detections
    for detection in mango_results.pred[0]:
        if detection[4] >= 0.15:  # Confidence threshold
            x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
            class_id = int(detection[-1])
            if mango_results.names[class_id].lower() == 'mango':
                mango_detected = True
                mango_message = "Mango Detected"  # Update global variable
                # Draw bounding box for mango detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for "Mango"
                cv2.putText(frame, "Mango Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                break

    if mango_detected:
        # Run ripeness model detection
        ripeness_results = ripeness_model(img_rgb)
        for detection in ripeness_results.pred[0]:
            if detection[4] >= 0.15:
                x1, y1, x2, y2 = map(int, detection[:4])
                ripeness_class_id = int(detection[-1])
                ripeness_label = ['Ripe-mango', 'Unripe-mango'][ripeness_class_id]
                # Draw bounding box for ripeness detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for ripeness
                cv2.putText(frame, f"Ripeness: {ripeness_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

        # Run disease model detection
        disease_results = disease_model(img_rgb)
        for detection in disease_results.pred[0]:
            if detection[4] >= 0.15:
                x1, y1, x2, y2 = map(int, detection[:4])
                disease_class_id = int(detection[-1])
                disease_label = disease_results.names[disease_class_id]  # Update global variable
                # Draw bounding box for disease detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for disease
                cv2.putText(frame, f"Disease: {disease_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                break

    return frame


def generate():
    global camera_active
    camera = cv2.VideoCapture(2)

    while camera_active:
        success, frame = camera.read()
        if not success:
            break

        # Resize and process frame
        resized_frame = cv2.resize(frame, (640, 480))
        processed_frame = process_frame(resized_frame)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/start_camera')
def start_camera():
    global camera_active
    camera_active = True
    return "Camera started", 200

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return "Camera stopped", 200


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def index():
    global mango_message, ripeness_label, disease_label
    cause, treatment, symptoms = "", "", ""
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
            
            time.sleep(0.0001)

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
                time.sleep(0.0001)

                # Disease detection
                disease_results = disease_model(filepath)
                disease_results.save()
                time.sleep(0.0001)

                # Get ripeness result
                if len(ripeness_results.pred) > 0 and len(ripeness_results.pred[0]) > 0:
                    ripeness_class_id = int(ripeness_results.pred[0][0][-1])
                    ripeness_label = ['Ripe-mango', 'Unripe-mango'][ripeness_class_id]
                else:
                    ripeness_label = "No ripeness detected."

                # Get disease result
                if len(disease_results.pred) > 0 and len(disease_results.pred[0]) > 0:
                    disease_class_id = int(disease_results.pred[0][0][-1])
                    disease_label = disease_results.names[disease_class_id]
                    if disease_label == 'Anthracnose' or 'anthracnose':
                        cause = disease['anthracnose']['cause']
                        symptoms = disease['anthracnose']['symptoms']
                        treatment = disease['anthracnose']['treatment']
                    elif disease_label == 'Lasiodiplodia' or 'lasiodiplodia':
                        cause = disease['ser']['cause']
                        symptoms = disease['ser']['symptoms']
                        treatment = disease['ser']['treatment']
                    elif disease_label == 'Aspergillus' or 'aspergillus':
                        cause = disease['bmr']['cause']
                        symptoms = disease['bmr']['symptoms']
                        treatment = disease['bmr']['treatment']
                else:
                    disease_label = "No disease detected."

                return render_template('index.html', 
                                        uploaded_image=filepath, 
                                        mango_result_image=destination_path, 
                                        ripeness_label=ripeness_label,
                                        disease_label=disease_label, 
                                        cause=cause, 
                                        symptoms=symptoms, 
                                        treatment=treatment,
                                        mango_message=mango_message)
            else:
                return render_template('index.html', 
                                       uploaded_image=filepath, 
                                       mango_result_image=destination_path, 
                                       mango_message=mango_message)

    return render_template('index.html', mango_message=mango_message, ripeness_label=ripeness_label, disease_label=disease_label, cause=cause, symptoms=symptoms, treatment=treatment)

@app.route('/resources')
def resources():
    return render_template('resources.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify, make_response
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import subprocess
import json
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
HISTORY_FILE = 'request_history.json'
os.makedirs('reports', exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}

model = YOLO("yolo11n.pt")

plt.style.use('seaborn-v0_8')

class PDFReport(FPDF):
    def header(self):
        self.add_font('Montserrat', '', 'montserrat_regular.ttf', uni=True)
        self.set_font('Montserrat', '', 14)
        self.cell(0, 10, 'Отчет по детекции жирафов', 0, 1, 'C')
    
    def footer(self):
        self.set_y(-15)
        self.add_font('Montserrat', '', 'montserrat_regular.ttf', uni=True)
        self.set_font('Montserrat', '', 14)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

def save_to_history(request_data, result_data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'request': request_data,
        'result': {
            'file_type': result_data['file_type'],
            'total_unique_giraffes': result_data['total_unique_giraffes'],
            'avg_confidence': result_data['avg_confidence'],
            'plots': result_data['plots']
        }
    }
    
    history.append(entry)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def generate_pdf_report(result_data):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Montserrat', '', 14)
    
    pdf.cell(0, 10, f"Тип файла: {result_data['file_type']}", 0, 1)
    pdf.cell(0, 10, f"Уникальных жирафов обнаружено: {result_data['total_unique_giraffes']}", 0, 1)
    pdf.cell(0, 10, f"Средняя уверенность: {result_data['avg_confidence']:.2f}", 0, 1)
    
    for plot_name, plot_data in result_data['plots'].items():
        plot_path = f"temp_{plot_name}.png"
        with open(plot_path, 'wb') as f:
            f.write(base64.b64decode(plot_data))
        
        pdf.add_page()
        pdf.set_font('Montserrat', '', 14)
        
        if plot_name == 'conf_hist':
            title = "Распределение уверенности модели"
        elif plot_name == 'box_sizes':
            title = "Размеры обнаруженных объектов"
        elif plot_name == 'frames':
            title = "Количество обнаружений по кадрам"
        else:
            title = plot_name
            
        pdf.cell(0, 10, title, 0, 1)
        pdf.image(plot_path, x=10, y=30, w=180)
        os.remove(plot_path)
    
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = os.path.join('reports', report_filename)
    pdf.output(report_path)
    
    return report_path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_giraffes(filepath):
    if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        return process_image(filepath)
    elif filepath.lower().endswith(('.mp4', '.mov')):
        return process_video(filepath)
    else:
        raise ValueError("Unsupported file format")
    
def generate_metrics_plots(detections, output_dir, filename_prefix):
    plots = {}
    
    if any('confidence' in det for det in detections):
        confidences = [det['confidence'] for det in detections if 'confidence' in det]
        plt.figure(figsize=(8, 4))
        plt.hist(confidences, bins=15, color='skyblue', edgecolor='black')
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        plots['conf_hist'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        img_buf.close()
    
    if any('bbox' in det for det in detections):
        boxes = [det['bbox'] for det in detections if 'bbox' in det]
        widths = [box[2] - box[0] for box in boxes]
        heights = [box[3] - box[1] for box in boxes]
        
        plt.figure(figsize=(8, 4))
        plt.scatter(widths, heights, alpha=0.6, color='green')
        plt.title('Bounding Box Sizes')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        plots['box_sizes'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        img_buf.close()
    
    if any(isinstance(det, dict) and 'frame' in det for det in detections):
        frames = [det['frame'] for det in detections if isinstance(det, dict) and 'frame' in det]
        counts = [len(det['detections']) if 'detections' in det else 0 for det in detections]
        
        plt.figure(figsize=(8, 4))
        plt.plot(frames, counts, 'o-', color='orange')
        plt.title('Detections per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Giraffes')
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        plots['frames'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        img_buf.close()
    
    return plots

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    
    results = model(image, classes=[23])
    
    detections = []
    all_confidences = []
    
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        all_confidences.append(confidence)
        detections.append({
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Giraffe {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, image)
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    plots = generate_metrics_plots(detections, RESULT_FOLDER, "img")
    
    return {
        "detections": detections,
        "result_path": result_path,
        "file_type": "image",
        "total_unique_giraffes": len(detections),
        "avg_confidence": avg_confidence,
        "plots": plots
    }

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_path = os.path.join(RESULT_FOLDER, "temp_" + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    giraffe_ids = set()
    all_confidences = []
    detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(
            frame,
            classes=[23],
            persist=True,
            tracker="bytetrack.yaml"
        )
        
        frame_detections = []
        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                confidence = float(box.conf[0])
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                giraffe_ids.add(track_id)
                all_confidences.append(confidence)
                frame_detections.append({
                    "id": track_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        detections.append({
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "detections": frame_detections
        })
        out.write(frame)
    
    cap.release()
    out.release()

    result_path = os.path.join(RESULT_FOLDER, "processed_" + os.path.basename(video_path))
    try:
        subprocess.run([
            'ffmpeg',
            '-y',
            '-i', temp_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            result_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video conversion failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    plots = generate_metrics_plots(detections, RESULT_FOLDER, "vid")
    
    return {
        "detections": detections,
        "result_path": result_path,
        "file_type": "video",
        "total_unique_giraffes": len(giraffe_ids),
        "avg_confidence": avg_confidence,
        "plots": plots
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                result = detect_giraffes(filepath)
                save_to_history({
                    'filename': filename,
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, result)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "File type not allowed"}), 400
    
    return render_template("index.html")

@app.route("/history")
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])
    
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    return jsonify(history)

@app.route("/generate_report/<timestamp>")
def get_report(timestamp):
    if not os.path.exists(HISTORY_FILE):
        return jsonify({"error": "History not found"}), 404
    
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    entry = next((item for item in history if item['timestamp'] == timestamp), None)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404
    
    report_path = generate_pdf_report(entry['result'])
    
    with open(report_path, 'rb') as f:
        pdf_data = f.read()
    
    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=report_{timestamp}.pdf'
    
    return response

if __name__ == "__main__":
    app.run(debug=True)
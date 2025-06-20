import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, DFL
from ultralytics.nn.modules.head import Detect
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.conv import Concat
from ml_model.detect import StarWarsDetector
import base64
from io import BytesIO
from PIL import Image

# Add safe globals for model loading
add_safe_globals([
    Upsample,
    MaxPool2d,
    Bottleneck,
    ModuleList,
    DetectionModel,
    Sequential,
    Conv,
    C2f,
    SPPF,
    Detect,
    Conv2d,
    BatchNorm2d,
    SiLU,
    Concat,
    DFL
])

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuración de la aplicación
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Asegurarse de que el directorio de uploads existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ruta al modelo
MODEL_PATH = os.path.join('ml_model', 'best.pt')

# Inicializar el detector
detector = StarWarsDetector(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = detector.detect(filepath)
            detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
            
            return jsonify({
                'success': True,
                'detections': detections
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            results = detector.detect(filepath)
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                detections.append({
                    'label': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
            # Generar imagen con las detecciones
            result_img = results.plot()
            pil_img = Image.fromarray(result_img)
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_data_url = f"data:image/jpeg;base64,{img_str}"
            return jsonify({
                'success': True,
                'image': img_data_url,
                'detections': detections
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 
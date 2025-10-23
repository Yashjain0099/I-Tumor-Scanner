from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import logging
import base64
from datetime import datetime
import json
from io import BytesIO
from PIL import Image, ImageEnhance
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
MODEL_PATH = "iris_tumor_cnn_model.keras"

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("history", exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH, compile=False)

# Define image size
IMG_SIZE = (224, 224)

def preprocess_image(img_path, brightness=1.0, contrast=1.0):
    """Preprocess the image for model prediction with enhancement options"""
    img = Image.open(img_path)
    
    # Apply enhancements if specified
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    # Resize and convert to array
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def save_to_history(filename, result, confidence, raw_value, image_data):
    """Save prediction to history"""
    try:
        history_file = "history/predictions.json"
        
        # Load existing history
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new entry
        entry = {
            "id": len(history) + 1,
            "filename": filename,
            "result": result,
            "confidence": confidence,
            "raw_value": raw_value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": image_data[:100] + "..." if len(image_data) > 100 else image_data  # Store truncated for file size
        }
        
        history.insert(0, entry)  # Add to beginning
        
        # Keep only last 50 predictions
        history = history[:50]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving to history: {str(e)}")
        return False

def get_history():
    """Retrieve prediction history"""
    try:
        history_file = "history/predictions.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        return []

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if files were uploaded
        if "files[]" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist("files[]")
        
        if not files or files[0].filename == "":
            return jsonify({"error": "No selected files"}), 400

        results = []
        
        try:
            # Get enhancement parameters
            brightness = float(request.form.get('brightness', 1.0))
            contrast = float(request.form.get('contrast', 1.0))
            
            for file in files:
                if file.filename == "":
                    continue
                    
                # Save the uploaded file
                filepath = os.path.join("uploads", file.filename)
                file.save(filepath)
                logger.info(f"Saved uploaded file: {filepath}")

                # Preprocess the image with enhancements
                img_array = preprocess_image(filepath, brightness, contrast)
                logger.info(f"Image shape after preprocessing: {img_array.shape}")

                # Make prediction
                prediction = model.predict(img_array)
                prediction_value = float(prediction[0][0])
                logger.info(f"Raw prediction value: {prediction_value}")

                # Threshold for classification
                threshold = 0.3
                result = "Tumorous" if prediction_value > threshold else "Healthy"
                
                confidence = prediction_value if result == "Tumorous" else (1 - prediction_value)
                confidence_percentage = f"{confidence * 100:.2f}%"
                
                logger.info(f"Final prediction: {result} with confidence {confidence_percentage}")
                
                # Get base64 encoded image 
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Save to history
                save_to_history(file.filename, result, confidence_percentage, 
                              f"{prediction_value:.4f}", img_data)
                
                # Add to results
                results.append({
                    "filename": file.filename,
                    "result": result,
                    "confidence": confidence_percentage,
                    "raw_value": f"{prediction_value:.4f}",
                    "image": img_data
                })
                
                # Remove the uploaded file
                os.remove(filepath)
            
            # Return JSON for AJAX or render template for form submission
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": True, "results": results})
            
            # For single file, render result page
            if len(results) == 1:
                return render_template("result.html", **results[0])
            
            # For multiple files, render batch results
            return render_template("batch_result.html", results=results)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": str(e)}), 500
            return f"Error processing image: {str(e)}", 500

    return render_template("index.html")

@app.route("/history")
def history():
    """Display prediction history"""
    predictions = get_history()
    return render_template("history.html", predictions=predictions)

@app.route("/api/history")
def api_history():
    """API endpoint for history"""
    predictions = get_history()
    return jsonify(predictions)

@app.route("/api/clear-history", methods=["POST"])
def clear_history():
    """Clear prediction history"""
    try:
        history_file = "history/predictions.json"
        if os.path.exists(history_file):
            os.remove(history_file)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export/<format>")
def export_results(format):
    """Export results in CSV or JSON format"""
    try:
        predictions = get_history()
        
        if format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["ID", "Filename", "Result", "Confidence", "Raw Value", "Timestamp"])
            
            for pred in predictions:
                writer.writerow([
                    pred.get("id", ""),
                    pred.get("filename", ""),
                    pred.get("result", ""),
                    pred.get("confidence", ""),
                    pred.get("raw_value", ""),
                    pred.get("timestamp", "")
                ])
            
            output.seek(0)
            return send_file(
                BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'iris_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        
        elif format == "json":
            return send_file(
                BytesIO(json.dumps(predictions, indent=2).encode()),
                mimetype='application/json',
                as_attachment=True,
                download_name=f'iris_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
        
        return jsonify({"error": "Invalid format"}), 400
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/about")
def about():
    """About page with educational information"""
    return render_template("about.html")

@app.route("/api/stats")
def get_stats():
    """Get statistics about predictions"""
    try:
        predictions = get_history()
        
        total = len(predictions)
        tumorous = sum(1 for p in predictions if p.get("result") == "Tumorous")
        healthy = total - tumorous
        
        stats = {
            "total": total,
            "tumorous": tumorous,
            "healthy": healthy,
            "tumorous_percentage": (tumorous / total * 100) if total > 0 else 0,
            "healthy_percentage": (healthy / total * 100) if total > 0 else 0
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
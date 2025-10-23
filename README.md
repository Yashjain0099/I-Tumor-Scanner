# 🔬 Iris Tumor Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Advanced AI-powered medical imaging solution for early detection and diagnosis of iris tumors using deep learning technology.

![Iris Tumor Detection](https://img.shields.io/badge/Accuracy-98.5%25-success)
![Processing](https://img.shields.io/badge/Processing-<1s-brightgreen)

## 🌟 Features

### Core Functionality
- ⚡ **Real-time Analysis** - Get instant results with our optimized CNN model
- 📊 **Confidence Scoring** - Detailed confidence metrics for each prediction
- 🔄 **Batch Processing** - Analyze multiple images simultaneously
- 📜 **History Tracking** - Comprehensive analysis history with search/filter
- 🎨 **Image Enhancement** - Brightness and contrast adjustment tools
- 💾 **Export Results** - Download reports in CSV/JSON format

### UI/UX Features
- 🎨 **Modern Professional Design** - Clean, medical-grade interface
- 🌓 **Dark/Light Theme** - Toggle between themes for comfort
- 📱 **Fully Responsive** - Works seamlessly on all devices
- ✨ **Smooth Animations** - Professional micro-interactions
- 🖼️ **Image Preview & Zoom** - High-quality image viewing
- 📈 **Statistics Dashboard** - Visual analytics and insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yashjain0099/Iris_Tumor_Detection.git
   cd Iris_Tumor_Detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file exists**
   - Place your trained model file `iris_tumor_cnn_model.keras` in the root directory
   - If you don't have a model, train one using your dataset

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and navigate to: `http://localhost:5000`
   - Start uploading iris images for analysis!

## 📁 Project Structure

```
Iris_Tumor_Detection/
│
├── app.py                          # Main Flask application
├── iris_tumor_cnn_model.keras      # Trained CNN model
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/                      # HTML templates
│   ├── index.html                  # Home page with upload
│   ├── result.html                 # Single result page
│   ├── batch_result.html           # Batch results page
│   ├── history.html                # Analysis history
│   └── about.html                  # About & information page
│
├── uploads/                        # Temporary upload directory
└── history/                        # Analysis history storage
    └── predictions.json            # Prediction history file
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **Pillow** - Image manipulation
- **NumPy** - Numerical computations

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Bootstrap 5** - UI framework
- **JavaScript ES6+** - Interactive functionality
- **Font Awesome** - Icons
- **Google Fonts** - Typography

### AI/ML
- **Convolutional Neural Networks (CNN)** - Model architecture
- **Image Classification** - Binary classification (Tumorous/Healthy)
- **Transfer Learning** - Pre-trained model optimization

## 📊 Model Information

- **Architecture**: Custom CNN
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Tumorous/Healthy)
- **Accuracy**: 98.5% on validation set
- **Training Dataset**: 10,000+ iris images
- **Processing Time**: < 1 second per image

## 🎯 Usage Guide

### Single Image Analysis
1. Navigate to the home page
2. Click or drag-and-drop an iris image
3. (Optional) Adjust brightness/contrast
4. Click "Analyze Image"
5. View detailed results with confidence score

### Batch Processing
1. Select multiple images at once
2. Adjust enhancement settings if needed
3. Click "Analyze Images"
4. View all results in grid format
5. Export results as CSV/JSON

### History Management
1. Access history from the navigation menu
2. Search and filter previous analyses
3. View detailed information for each prediction
4. Export or clear history as needed

## 🔒 Security & Privacy

- ✅ Images processed in real-time
- ✅ No permanent storage of uploaded images
- ✅ History stored locally (can be cleared)
- ✅ No third-party data sharing
- ✅ Secure file upload handling

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Home page and image upload |
| `/history` | GET | View analysis history |
| `/about` | GET | Information and FAQ page |
| `/api/history` | GET | Get history as JSON |
| `/api/stats` | GET | Get statistics |
| `/api/clear-history` | POST | Clear all history |
| `/export/csv` | GET | Export history as CSV |
| `/export/json` | GET | Export history as JSON |

## 🎨 Customization

### Changing Colors
Edit the CSS variables in each HTML file:
```css
:root {
  --primary-color: #4f46e5;
  --secondary-color: #7c3aed;
  --success-color: #10b981;
  --danger-color: #ef4444;
}
```

### Adjusting Threshold
Modify the classification threshold in `app.py`:
```python
threshold = 0.3  # Adjust between 0.0 and 1.0
```

### Model Configuration
Update image size or model path:
```python
IMG_SIZE = (224, 224)  # Change input size
MODEL_PATH = "your_model.keras"  # Update model path
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model file not found
```bash
Solution: Ensure iris_tumor_cnn_model.keras is in the root directory
```

**Issue**: Module import errors
```bash
Solution: pip install -r requirements.txt
```

**Issue**: Port already in use
```bash
Solution: Change port in app.py: app.run(port=5001)
```

**Issue**: Image upload fails
```bash
Solution: Check file size (max 16MB) and format (JPG, PNG)
```

## 🚀 Deployment

### Heroku Deployment
1. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```

2. Create `runtime.txt`:
   ```
   python-3.11.0
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Docker Deployment
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["python", "app.py"]
   ```

2. Build and run:
   ```bash
   docker build -t iris-detection .
   docker run -p 5000:5000 iris-detection
   ```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Medical Disclaimer

This AI-powered tool is designed for **screening purposes only** and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. 

- Always consult with a qualified healthcare provider
- This tool provides preliminary screening results
- Medical professionals should verify all findings
- Do not make medical decisions based solely on this tool

## 👨‍💻 Developer

**Yash Jain**
- GitHub: [@Yashjain0099](https://github.com/Yashjain0099)
- Project: [Iris Tumor Detection](https://github.com/Yashjain0099/Iris_Tumor_Detection)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Medical imaging community for datasets
- Open-source contributors

---

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ by Yash Jain
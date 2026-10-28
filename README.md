# 🌍 Disaster Management and Response System Using Machine Learning



## 🧠 Overview

This project presents an **AI-powered Disaster Management and Response System** that assists authorities and citizens during natural calamities such as floods, earthquakes, wildfires, and cyclones.  
It integrates **Machine Learning**, **Flask-based web infrastructure**, and **automation APIs** to perform:

- Real-time **disaster classification**  
- **Alert generation** and notification  
- **Rescuer assignment** and dashboard-based coordination  

The system aims to reduce manual intervention, minimize response time, and enhance coordination between citizens, rescuers, and administrators.

---

## ⚙️ System Architecture

User Input / Sensor Data
│
▼
[ Data Preprocessing & ML Model ]
│
▼
Prediction (Disaster Type & Severity)
│
▼
Automated Alert Generation & Rescuer Assignment
│
▼
Flask Web Dashboard (User / Rescuer / Admin)
│
▼
Database & Reporting (MongoDB)

yaml
Copy code

---

## 🏗️ Project Structure

Disaster-Management/
│
├── app/ # Flask web app files
├── models/ # Trained ML models (e.g., MobileNetV2 + LSTM)
├── data/ # Sample or source data links
├── notebooks/ # Jupyter notebooks for training/testing
├── static/ # CSS, JS, images for the web app
├── templates/ # HTML templates (User/Admin dashboards)
├── docs/ # Architecture diagrams, screenshots
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🧩 Key Features

✅ **Disaster Type Classification** — ML model trained on multi-type disaster datasets (flood, fire, earthquake, cyclone)  
✅ **Automated Alert Generation** — Uses Python APIs and automation to notify rescuers and citizens  
✅ **Flask-based Role System** — Three dashboards (User, Rescuer, Admin) with real-time updates  
✅ **Integration with MongoDB** — Efficient data storage for disaster reports and rescuer details  
✅ **Transfer Learning** — Improved accuracy using pretrained MobileNetV2  
✅ **Generative AI Integration** — Chatbot and voice alert support (planned extension)

---

## 📊 Machine Learning Model

| Model | Accuracy | Recall | F1-Score |
|--------|-----------|---------|-----------|
| Decision Tree | 85.2% | 84.0% | 83.8% |
| CNN + LSTM | 89.7% | 88.4% | 88.0% |
| **MobileNetV2 + LSTM (Final)** | **92.1%** | **91.3%** | **90.9%** |

**Training Details**
- Frameworks: TensorFlow, scikit-learn  
- Optimizer: Adam (lr = 0.001)  
- Loss Function: Categorical Cross Entropy  
- Batch Size: 32, Epochs: 50  
- Hardware: NVIDIA RTX 3060 GPU  

---

## 🧮 Data Sources

1. [Kaggle Disaster Response Dataset](https://www.kaggle.com/competitions/disaster-response-messages)
2. [NASA Earth Observatory Datasets](https://earthdata.nasa.gov/)
3. [Disaster Response Data Pipeline (Open Source)](https://www.figure-eight.com/data/)
4. Real-time data collected through Flask reporting interface.

---

## 🌐 Web Application

The **Flask web interface** provides three dashboards:
- **User Dashboard** → Submit disaster reports and receive alerts  
- **Rescuer Dashboard** → Receive automated assignments and update response status  
- **Admin Dashboard** → Monitor all operations in real time  

---

## 🧰 Installation and Usage

### 🔧 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
▶️ Run the Flask Application
python app.py
The application will be live at:
👉 http://127.0.0.1:5000/

🧠 Train/Run the Model
python train_model.py
python predict_disaster.py
🔒 Ethical & Privacy Considerations
All user and location data are encrypted to maintain privacy.

The model ensures fairness across regions and conditions.

Human validation is preserved in critical operations to prevent AI misuse.

📈 Future Enhancements
🌩 IoT Integration — Connect to sensor data and satellite feeds

📱 Mobile & Edge Deployment — Using TensorFlow Lite for real-time field use

🤖 NLP & Chatbot Support — Voice-based disaster reporting and AI chat assistance

☁️ Cloud Scaling — AWS Lambda and load balancing for concurrent users

🧾 Citation
If you use this project or its concepts in your research, please cite:

Aayush Goyal (2025). “Disaster Management and Response System Using Machine Learning.”
Maharaja Agrasen Institute of Technology, Department of Information Technology, New Delhi, India.
GitHub: https://github.com/aayushgoyal2503/Disaster-Management

🧩 License
This project is licensed under the MIT License — you are free to use and modify it with proper attribution.
See the LICENSE file for details.

💬 Contact
For questions, collaborations, or deployment discussions:
📧 aayush.10463@gmail.com

⭐ If you found this project helpful, please star the repository on GitHub!








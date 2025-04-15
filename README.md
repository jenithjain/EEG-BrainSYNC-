# 🧠 BrainSYNC: Intelligent EEG-Based Cognitive Disorder Prediction System

## 🚀 Overview

**BrainSYNC** is an AI-powered system designed to analyze EEG data and provide intelligent medical insights for early detection and monitoring of cognitive disorders such as depression, anxiety, epilepsy, and addiction. The system processes EEG data (both from structured CSV and parsed PDF reports), extracts meaningful neurophysiological features, performs deep learning-based classification, and integrates a GPT-powered doctor agent to generate personalized medical recommendations.

---

## 🧩 Problem Statement

Cognitive disorders often remain undiagnosed until they manifest as serious behavioral or psychological issues. Traditional EEG analysis is manual, subjective, and time-consuming. **BrainSYNC** aims to solve this by building an end-to-end pipeline that:

- Automatically extracts features from EEG reports
- Learns spatial, frequency, and temporal EEG patterns
- Predicts possible neurological disorders
- Offers doctor-like feedback through a GPT-based agent

---

## 📊 Dataset Used

- **Source**: [EEG.machinelearing_data_BRMH.csv](https://github.com/user-attachments/files/19758673/EEG.machinelearing_data_BRMH.csv)
- **Contents**: EEG feature dataset with band powers, coherence measures, and associated labels
- **Columns Include**:
  - **EEG Channels**: Band powers (delta, theta, alpha, beta, gamma) per channel
  - **Labels**: `main.disorder`, `specific.disorder`, and demographic details
  - **Target**: Predicting specific neurological disorders from EEG signals

---

## 🧠 Model Architecture

A custom **Deep Neural Network (DNN)** was built using **PyTorch** to learn complex brain activity patterns from EEG features.

### 🧬 Network Learning Process

- **Layer 1 (256 neurons)**:
  - Learns basic EEG waveforms and simple spatial patterns
- **Layer 2 (128 neurons)**:
  - Combines wave features, starts detecting disorder-relevant markers
- **Layer 3 (64 neurons)**:
  - Generates abstract neural representations
- **Output Layer**:
  - Classifies the input into specific cognitive disorders
  - Outputs confidence scores for medical decision support

---

## 🔍 Feature Engineering

### 🧠 Spatial Analysis
- Maps activation patterns across brain regions
- Identifies regional abnormalities in activity

### 🎵 Frequency Analysis
- Calculates band powers (Delta, Theta, Alpha, Beta, Gamma)
- Detects abnormal patterns and dominance in frequency bands

### ⏱️ Temporal Analysis
- Examines time-based activity variations
- Detects instability or fluctuations in brain states

---

## 🧑‍⚕️ Doctor Agent Integration

A GPT-powered **Doctor Agent** analyzes the EEG classification results and:

- Generates treatment suggestions based on disorder type
- Considers patient history and severity
- Recommends follow-up tests or behavioral therapies
- Explains reasoning in simple, doctor-style language

---

## 📦 Folder Structure
# 📁 BrainSYNC Project Directory Structure

```bash
BrainSYNC/
├── app/
│   ├── pdf_parser.py             # PDF report processing and EEG data extraction
│   └── streamlit_app.py          # (Optional) Streamlit interface for interactive EEG analysis
│
├── api/
│   ├── routers/
│   │   ├── predictions.py        # API endpoints for EEG-based disorder prediction
│   │   └── treatments.py         # API endpoints for treatment recommendations and simulations
│   └── models/
│       └── schemas.py            # Pydantic schemas for request/response models
│
├── predict/
│   └── predict_cognitive.py      # Core EEG analysis, preprocessing, and DNN-based prediction logic
│
├── models/
│   ├── cognitive_model.pt        # Main trained PyTorch model for multi-disorder classification
│   ├── best_model.pt             # Best-performing model (validated on EEG test data)
│   └── fine_tuned_model.pt       # Fine-tuned model optimized for specific neurological conditions
│
├── data/
│   ├── reports/                  # Structured JSON files extracted from EEG PDF reports
│   └── test_eeg.pt               # Sample EEG test data tensor for model validation
│
├── outputs/
│   ├── eeg_analysis.png          # Visual heatmaps and plots of EEG signal distribution
│   ├── diagnosis_result.png      # Classification output visualized
│   └── diagnosis_explanation.png # Visualization of GPT doctor agent’s reasoning
│
├── run_api.py                    # Entry point for running FastAPI backend (if needed)
├── requirements.txt              # List of dependencies to install for BrainSYNC
└── README.md                     # Project documentation and instructions



---

## 🔬 Results

- **Accuracy**: > 90% in predicting known disorders from EEG data
- **Model Confidence**: Scored probabilities for each prediction
- **Agent Performance**: Provided clinically relevant, readable advice

---

## 🧠 Supported Disorders

- Depression
- Epilepsy
- Cognitive impairment
- Addiction
- Anxiety disorders

---

## 🛠 Tools & Technologies

- **Language**: Python
- **Deep Learning**: PyTorch
- **Data Parsing**: pandas, PyPDF2
- **AI Agent**: GPT-based LLM (OpenAI/Gemini)

---

## 🔮 Future Enhancements

- Real-time EEG signal processing
- Integration with wearable EEG devices
- Personalized patient dashboards
- Multi-modal brain-health analytics (EEG + MRI + lifestyle)
- Outcome tracking and longitudinal analysis

---

## ✍️ Sample Output Format

```json
{
  "report_id": "EEG_20250413_174500",
  "patient_info": {
    "id": "P123",
    "diagnosis": "Major Depressive Disorder",
    "confidence": 0.94
  },
  "eeg_analysis": {
    "band_powers": {
      "delta": "...",
      "theta": "...",
      "alpha": "...",
      "beta": "...",
      "gamma": "..."
    },
    "coherence": {
      "F3-F4": "...",
      "Pz-Oz": "..."
    }
  },
  "doctor_recommendations": {
    "treatment": "Start CBT + SSRIs. Review in 2 weeks.",
    "follow_up": "Schedule a sleep pattern EEG & psych eval"
  }
}

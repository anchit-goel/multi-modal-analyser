# 🎙️ Audio Spoof / Deepfake Detection System

A from-scratch audio deepfake (spoof) detection system using CNN and LCNN models, designed for real-time inference with explainable risk scoring.

---

## 🚀 Overview

This project detects whether an audio sample is:

- ✅ Genuine (Real human speech)  
- ❌ Spoof (TTS / Voice Conversion)  

It uses:
- Log-Mel Spectrogram features → CNN model  
- LFCC features → LCNN model  
- Ensemble fusion → final decision  

---

## 🧠 Key Features

- From-scratch model implementation  
- CNN + LCNN ensemble  
- Feature engineering (LogMel + LFCC)  
- Risk scoring system  
- Explainable outputs  
- Real-time inference (~30ms per audio chunk)  

---

## 📁 Project Structure

audio_spoof_system/
│
├── models/
│   ├── cnn_best_clean.pth
│   └── lcnn_best_clean.pth
│
├── inference.py  
├── models_architecture.py  
├── requirements.txt  
└── README.md  

---


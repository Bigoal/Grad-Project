# Grad-Project
AI-enhanced cloud Web Application Firewall that detects malicious HTTP requests using TF-IDF + machine learning on Google Cloud.

This repository contains my graduation project: an AI-Enhanced Cloud-Based Web Application Firewall (WAF). The system analyzes HTTP request traffic, converts requests into TF-IDF feature vectors (1,464,061 requests processed), and uses machine-learning classifiers to detect malicious requests. The solution was developed with Google Cloud AI services to enable scalable, near real-time detection and evaluation.

Key highlights:

- Vectorized ~1.46M HTTP requests (1,413,061 benign; 51,000 malicious) using TF-IDF.
- Trained and evaluated multiple machine-learning classifiers to distinguish malicious from benign traffic.
- Designed for cloud deployment to scale with traffic and support near real-time inference.
- Includes reproducible notebooks, model training code, and deployment examples.

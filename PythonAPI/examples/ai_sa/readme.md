# Human Situation Awareness Guided Risk Predictor

This project implements a vision-based risk prediction model.  
It takes a sequence of three consecutive frames as input and outputs a risk score along with an attention heatmap for interpretability.

## Features
- **Sequence Input**: Processes three consecutive RGB frames.
- **Risk Prediction**: Outputs a single scalar risk score.
- **Attention Visualization**: Generates a 224×224 attention heatmap.

## Project Structure
├── ai_sa_model.py # Model architecture 
├── inference.py # Inference function
├── main.py # Entry point with example usage
├── util.py # Utility functions (I/O, visualization, GPU check)
├── model_ckpt/ # Model weights
├── test_frame/ # Test images (frame0.png, frame1.png, frame2.png)
└── out/ # Output directory for attention maps



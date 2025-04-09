# video-sdk-asssignmet
SpeechT5 Fine-tuning for Indic Languages
This repository contains code for fine-tuning Microsoft's SpeechT5 text-to-speech model on Indic languages using the AI4Bharat IndicVoices-R dataset. The implementation enables low-resource adaptation of a state-of-the-art TTS model for Hindi, Tamil, Bengali, Gujarati, Telugu, Marathi, and other Indian languages.
Features

🔊 Fine-tune SpeechT5 on Indian language speech datasets
🗣️ Support for multiple Indic languages
📊 Training with smaller dataset sizes for faster experimentation
💾 Google Drive integration for saving checkpoints and models
🎯 Memory-efficient implementation with streaming datasets
📝 Audio preprocessing and normalization
📈 Training with mixed precision and checkpoint resumption

Prerequisites

Python 3.7+
PyTorch and TorchAudio
Transformers library (Hugging Face)
Google Colab environment (for Google Drive integration)
GPU for faster training

Installation
bashpip install torch torchaudio transformers datasets librosa soundfile matplotlib pandas tqdm numpy
Dataset
This code uses the AI4Bharat IndicVoices-R dataset, which contains high-quality recordings of Indic language speech with corresponding text transcriptions. The dataset is automatically downloaded through the Hugging Face datasets library.



Usage
Basic Usage
![image](https://github.com/user-attachments/assets/47fd5ff5-5a80-4132-bf07-87ad55634bed)



# Train the model
![image](https://github.com/user-attachments/assets/82560d73-235c-49ca-88eb-efa1882754c8)

Key Components
Model Architecture
This implementation uses:

SpeechT5Processor: For tokenizing text input
SpeechT5ForTextToSpeech: The main TTS model
SpeechT5HifiGan: High-quality vocoder for converting model output to audio

Data Pipeline

Dataset Loading: Streams data from the IndicVoices-R dataset
Text Preprocessing: Cleans and normalizes text input
Audio Preprocessing: Normalizes and trims silence from audio
Feature Extraction: Converts inputs to model-compatible format
Collation: Handles batching and padding for efficient training

Training Loop
The training process includes:

Mixed precision training (FP16)
Gradient accumulation for stable training
Checkpoint saving and resumption
Evaluation during training


# Generate sample speech
test_text = "नमस्ते, आप कैसे हैं?"  # Example for Hindi
speech = generate_speech_sample(test_text, model, processor, vocoder)
Supported Languages
The code supports the following Indic languages:
Hindi
Tamil
Gujarati
Bengali
Telugu
Marathi



Performance Considerations

Training on a full dataset requires significant computational resources
The implementation supports smaller dataset sizes for experimentation
GPU acceleration is highly recommended
Memory usage is optimized through streaming datasets

Troubleshooting

If you encounter "CUDA out of memory" errors, reduce batch size or max_samples
If language loading fails, try alternative language codes
For dataset access issues, ensure proper authentication with Hugging Face

License
This code is provided under the MIT License. The SpeechT5 model and IndicVoices-R dataset are subject to their respective licenses.
Acknowledgments

Microsoft SpeechT5 for the base TTS model
AI4Bharat for the IndicVoices-R dataset
Hugging Face for the Transformers and Datasets libraries

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


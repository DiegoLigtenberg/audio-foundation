# GPT creation, non chat yet so no reinforcement learning human feedback

# audio-foundation
 experimentation for big dataset audio models

# 1.0 Youtube scraper
- Create a list of 1 million + url links of youtube videos (somehow select some heuristics to check for quality)
  - videos > 30 sec
  - videos < 60 min
  - video's with average sound frequency > x 

## 1.1 Dataloader
- Create a dataloder that captures N=10k/1mil+ sampled youtube videos
- from these 10k videos, load batchsize in memory, convert them to from whatever to waveform, convert waveform to mel spectrogram


# 2.0 model
- put the batched mel spectrogram through model

# 3.0 inference
- use a vocoder to transform mel spectrogram to high fidelty sound








################################### ################################### ################################### ################################### ################################### 
# Audio Foundation - GPT-like Architecture for Audio Modeling

This document outlines a plan for building a large audio-based GPT model, initially for tasks like audio generation, completion, or transformation, focusing on Mel spectrograms as the core input representation. The model is intended to train on large-scale datasets (such as YouTube) and generate high-quality audio outputs.

---

## 1.0 Data Collection

### 1.1 YouTube Scraper
- **Objective**: Collect a dataset of YouTube videos to train the model. The dataset must have diverse audio characteristics and contain videos long enough to provide meaningful audio content.
- **Selection Criteria**:
  - **Video Length**: Videos must be between **30 seconds and 60 minutes** to ensure sufficient audio content.
  - **Audio Quality**: Select videos with an **average sound frequency** greater than a threshold \( x \), ensuring that the audio content has enough diversity and richness.
  - **Additional Filters**:
    - Videos with speech or music (e.g., using metadata or simple classification models).
    - Remove videos with noisy, low-quality audio (e.g., background hum, distortion).

### 1.2 YouTube Data Scraping Pipeline
- **Input**: URL list of 1 million+ YouTube videos.
- **Output**: Processed videos, with timestamps and audio, ready for spectrogram extraction.
- **Procedure**:
  - **Extract Video URL List**: Scrape 1M+ YouTube URLs using YouTube API or a custom scraper.
  - **Download Audio**: Use `youtube-dl` or equivalent tool to download audio tracks from the selected videos.
  - **Preprocessing**:
    - Convert audio to **stereo** if needed.
    - Normalize audio quality (e.g., adjust volume, sample rate).
  
---

## 2.0 Data Preprocessing

### 2.1 Audio to Mel Spectrogram Conversion
- **Objective**: Convert raw audio data into Mel spectrograms for model input.
- **Pipeline**:
  - **Batch Processing**: Process 10k–1M videos in batches.
  - **Audio Format Conversion**: Convert the raw audio waveform into Mel spectrograms, which will act as the input representation for the model.
  - **Chunking**: Divide the audio into fixed-length chunks (e.g., 3-second segments) that can be processed in parallel.
  - **Mel Spectrogram Parameters**:
    - **Sample rate**: Choose an optimal sample rate (e.g., 22kHz or 44kHz).
    - **Window Size and Hop Length**: Define the FFT window size (e.g., 1024) and hop length (e.g., 512) for frame extraction.
    - **Number of Mel Bands**: Use 128 or more Mel frequency bands for high-frequency resolution.

- **Tools**: 
  - Python libraries such as `Librosa` or `Torchaudio` for audio processing.
  - `librosa.feature.melspectrogram` for Mel spectrogram conversion.
  - **Parallelization**: Consider using `multiprocessing` or `Dask` for large-scale preprocessing.

---

## 3.0 Model Architecture

### 3.1 Transformer-based Model
- **Objective**: Use a transformer architecture to model sequences of Mel spectrogram frames.
- **Model Input**: Batched Mel spectrograms of audio chunks (e.g., 3-second segments).
- **Model Output**: Predicted Mel spectrograms for the next time segment (e.g., 0.25–1.0 second).
  
#### Transformer Architecture Overview:
- **Encoder**: Process each Mel spectrogram chunk in parallel.
- **Decoder**: Predict the next Mel spectrogram frame using self-attention.
- **Position Encoding**: Use sinusoidal or learnable position encodings to account for the sequential nature of audio.
- **Training Objective**: Similar to NLP models, the model predicts the next segment (token) based on previous segments.

### 3.2 Training Strategy
- **Data Augmentation**: To improve robustness, augment the spectrograms:
  - **Time stretching**: Slightly modify the speed of audio segments.
  - **Pitch shifting**: Modify the pitch without affecting timing.
  - **Noise injection**: Add slight noise to simulate real-world audio conditions.
  
- **Loss Function**: Use Mean Squared Error (MSE) or more complex perceptual loss (e.g., using pretrained VGG for audio) to improve audio quality.

---

## 4.0 Vocoder (Audio Synthesis)

### 4.1 Vocoder for High-Fidelity Audio Generation
- **Objective**: Convert predicted Mel spectrograms back into high-fidelity audio using a vocoder.
- **Model**: Use an advanced vocoder such as **WaveGlow**, **HiFi-GAN**, or **MelGAN**.
  - These models are capable of converting Mel spectrograms into clean, high-fidelity audio.
  
- **Procedure**:
  - Input: Predicted Mel spectrogram (output of transformer model).
  - Output: High-quality waveform (audio file).

- **Considerations**:
  - Fine-tune the vocoder on the specific dataset for improved performance on YouTube-style audio.

---

## 5.0 Inference

### 5.1 Real-Time Inference Pipeline
- **Objective**: Generate high-fidelity audio from new Mel spectrograms in real-time or in batch.
- **Process**:
  - Feed a batch of spectrogram predictions into the trained vocoder.
  - Output the synthesized waveform in real-time.

### 5.2 Evaluation
- **Objective**: Measure the quality of the generated audio and model performance.
  - Use **Objective Metrics** (e.g., MSE, Spectral Distortion) to measure quality.
  - Use **Human Evaluation** for subjective assessment of audio fidelity.

- **Example Evaluation Criteria**:
  - **Realism**: Does the generated audio sound natural?
  - **Diversity**: Can the model generate a variety of sound types (speech, music, etc.)?
  - **Coherence**: Does the audio maintain temporal consistency (smooth transitions)?

---

## 6.0 Scaling and Improvements

### 6.1 Training on Larger Datasets
- **Objective**: Scale the model to handle an even larger dataset, potentially across multiple machines or GPUs.
  - Use **distributed training** techniques (e.g., Horovod or DeepSpeed).
  - Consider **multi-modal inputs** (e.g., combining spectrograms with text or metadata).
  
### 6.2 Fine-Tuning for Specific Tasks
- After training on the large YouTube dataset, fine-tune the model for specific tasks like:
  - **Speech-to-Text**: Convert speech audio to transcribed text.
  - **Audio Generation**: Generate new audio based on text prompts or other conditions.

### 6.3 Audio Inpainting (Optional)
- **Objective**: Improve audio generation by using an inpainting model that fills in missing parts of an audio segment (e.g., reconstruct missing parts of a speech signal).
  - This can be achieved by predicting missing portions of the spectrogram and using it in conjunction with the transformer.

---

## 7.0 Conclusion

The outlined plan aims to create a scalable GPT-like model for audio, which learns from massive amounts of YouTube audio data to generate high-quality, coherent sound. By using Mel spectrograms as input, a transformer-based architecture for sequence modeling, and a vocoder for audio synthesis, the system should be capable of generating realistic audio from complex input sequences.

---


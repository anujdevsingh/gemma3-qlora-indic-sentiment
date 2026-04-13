# Multilingual Speech Recognition with Whisper Large-V3

Automatic Speech Recognition (ASR) system for transcribing multilingual audio in **English**, **Hindi**, and **Tamil** using [OpenAI Whisper Large-V3](https://huggingface.co/openai/whisper-large-v3). Built for the NPPE-2 Kaggle competition on multilingual speech recognition.

**Author:** Anuj Dev Singh (21F3000028)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Configuration](#configuration)
- [Pipeline Details](#pipeline-details)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

This project implements a zero-shot multilingual ASR pipeline that transcribes speech across three Indian languages without any fine-tuning. It leverages the pretrained Whisper Large-V3 model (1.5B parameters, trained on 680K hours of multilingual audio) with beam search decoding to achieve competitive Word Error Rate (WER) scores on the Kaggle evaluation benchmark.

The system automatically detects the spoken language from the audio signal and transcribes in the **native script** (Latin for English, Devanagari for Hindi, Tamil script for Tamil) rather than translating everything to English.

## Key Features

- **Trilingual transcription** -- English, Hindi, and Tamil from a single unified model
- **Automatic language detection** -- no manual language labels required at inference time
- **Native script output** -- transcriptions preserve the original writing system (Devanagari, Tamil script)
- **Beam search decoding** -- 5-beam search for higher transcription accuracy
- **Zero-shot inference** -- no fine-tuning needed; works out of the box on a Kaggle T4 GPU

## Architecture

```
Audio Input (.wav, 16 kHz)
        |
        v
+-------------------+
| Feature Extractor |  WhisperFeatureExtractor
| (Log-Mel Spectrogram) |
+-------------------+
        |
        v
+-------------------+
| Language Detection |  model.detect_language()
+-------------------+
        |
        v
+-------------------+
| Encoder-Decoder   |  Whisper Large-V3 (1.5B params)
| Transformer       |  Beam search (num_beams=5)
+-------------------+
        |
        v
+-------------------+
| Token Decoding    |  WhisperProcessor.batch_decode()
+-------------------+
        |
        v
  Transcription Text
```

## Dataset

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 2,000 | Audio + ground-truth transcriptions |
| Test | 100 | Audio only (predictions submitted to Kaggle) |

**Language distribution (training set):**

| Language | Samples | Percentage |
|----------|---------|------------|
| English | 998 | 49.9% |
| Hindi | 508 | 25.4% |
| Tamil | 494 | 24.7% |

- **Audio format:** WAV, 16 kHz sample rate
- **Average duration:** ~7.23 seconds per sample
- **Source:** [Kaggle - Multilingual Speech Recognition](https://www.kaggle.com/competitions/multilingual-speech-recognition)

## Results

| Metric | Value |
|--------|-------|
| Validation WER (20-sample subset) | **0.2316** (23.16%) |
| Test predictions generated | 100 |

> WER (Word Error Rate) measures the edit distance between predicted and reference transcriptions. Lower is better.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA Tesla T4)
- [HuggingFace account](https://huggingface.co/) with an access token (for gated model downloads)

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install jiwer librosa soundfile evaluate
```

### Running the Notebook

1. **On Kaggle (recommended):**
   - Upload or fork the notebook on [Kaggle](https://www.kaggle.com/)
   - Add the competition dataset: `multilingual-speech-recognition`
   - Add your HuggingFace token as a Kaggle secret named `HF_TOKEN`
   - Enable GPU (T4) in notebook settings
   - Click **Run All**

2. **Locally:**
   - Download the competition data and place it under a directory matching the expected paths
   - Update `Config.INPUT_DIR` and `Config.OUTPUT_DIR` in the notebook
   - Run:
     ```bash
     jupyter notebook 21f3000028-nppe-2.ipynb
     ```

## Configuration

All hyperparameters are centralized in the `Config` class:

```python
class Config:
    INPUT_DIR     = Path("/kaggle/input")
    OUTPUT_DIR    = Path("/kaggle/working")
    MODEL_SAVE    = OUTPUT_DIR / "whisper-finetuned"
    SR            = 16000          # Audio sample rate (Hz)
    MODEL_NAME    = "openai/whisper-large-v3"
    EPOCHS        = 3              # (unused -- inference only)
    BATCH_SIZE    = 4              # (unused -- inference only)
    GRAD_ACCUM    = 4              # (unused -- inference only)
    LEARNING_RATE = 1e-5           # (unused -- inference only)
    WARMUP_STEPS  = 50             # (unused -- inference only)
    MAX_LABEL_LEN = 448
    SEED          = 42
```

**Inference parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_new_tokens` | 440 | Maximum output sequence length |
| `num_beams` | 5 | Beam search width for decoding |
| `task` | `"transcribe"` | Transcription mode (not translation) |
| `return_timestamps` | `True` | Enable timestamp-aware generation |

## Pipeline Details

The notebook is organized into the following stages:

| Stage | Section | Description |
|-------|---------|-------------|
| 1 | Environment Setup | CUDA configuration, package installation |
| 2 | Imports | Load all required libraries |
| 3 | Authentication | HuggingFace token login via Kaggle secrets |
| 4 | Configuration | Centralized hyperparameters |
| 5-6 | Data Loading | Locate and load train/test CSVs and audio directories |
| 7 | Language Analysis | Script-based language detection on training labels |
| 8 | Audio Verification | Validate audio files load correctly at 16 kHz |
| 9 | Model Loading | Download and initialize Whisper Large-V3 |
| 10 | Directory Resolution | Map train/test audio to correct filesystem paths |
| 11 | Transcription | Core inference function with beam search |
| 12 | Validation | WER evaluation on a 20-sample subset |
| 13 | Test Prediction | Generate transcriptions for all 100 test samples |
| 14 | Submission | Export predictions to `submission.csv` |

## Design Decisions

### Why inference-only (no fine-tuning)?

1. **Strong baseline performance** -- Whisper Large-V3 already achieves competitive WER on English, Hindi, and Tamil out of the box
2. **GPU memory constraints** -- fine-tuning 1.5B parameters exceeds the 16 GB VRAM available on Kaggle's Tesla T4
3. **Small dataset risk** -- with only 2,000 training samples, fine-tuning risks overfitting without meaningful generalization gains
4. **Smaller models degrade** -- experiments with Whisper Small/Medium showed significantly worse multilingual performance

### Why beam search?

Beam search (`num_beams=5`) explores multiple decoding paths simultaneously, reducing greedy errors in multilingual transcription where the model may initially assign high probability to a wrong script or phoneme.

## Project Structure

```
gemma3-qlora-indic-sentiment/
|-- 21f3000028-nppe-2.ipynb    # Main notebook (end-to-end pipeline)
|-- README.md                  # This file
|-- .gitattributes             # Git LFS / attributes configuration
```

**Output (generated at runtime):**

```
/kaggle/working/
|-- submission.csv             # Final predictions (audio, text)
```

## Technologies Used

| Category | Tool |
|----------|------|
| Deep Learning | [PyTorch](https://pytorch.org/) |
| ASR Model | [OpenAI Whisper Large-V3](https://huggingface.co/openai/whisper-large-v3) (1.5B params) |
| Model Hub | [HuggingFace Transformers](https://huggingface.co/docs/transformers) |
| Audio Processing | [librosa](https://librosa.org/), [soundfile](https://pysoundfile.readthedocs.io/) |
| Evaluation | [jiwer](https://github.com/jitsi/jiwer) (Word Error Rate) |
| Notebook Environment | Jupyter / Kaggle Notebooks |
| Hardware | NVIDIA Tesla T4 (16 GB VRAM) |

## License

This project is for educational and academic purposes as part of the NPPE-2 coursework assignment.

# Multilingual Speech Recognition with Whisper Large-V3

Zero-shot multilingual ASR system for transcribing speech in **English**, **Hindi**, and **Tamil** using OpenAI's Whisper Large-V3 model — built for the NPPE-2 Kaggle competition.

**Author:** Anuj Dev Singh (21F3000028)

---

## What This Project Does

Takes raw audio files and transcribes them in their native scripts — Devanagari for Hindi, Tamil script for Tamil, Latin for English — without any fine-tuning or manual language labels at inference time.

## Tech Stack

| Component | Tool |
|---|---|
| ASR Model | OpenAI Whisper Large-V3 (1.5B params) |
| Framework | HuggingFace Transformers |
| Audio Processing | librosa, soundfile |
| Evaluation | jiwer (Word Error Rate) |
| Runtime | Kaggle Notebooks (T4 GPU) |

## Approach

- **Zero-shot inference** — Whisper Large-V3 is pretrained on 680K hours of multilingual audio and achieves strong out-of-the-box WER on all 3 target languages
- **Beam search decoding** (num_beams=5) for higher transcription accuracy over greedy decoding
- **Auto language detection** — no manual language labels needed at inference time
- **Native script output** — preserves original writing systems rather than translating to English
- **No fine-tuning** — 1.5B parameters exceeds T4 VRAM; fine-tuning smaller variants degraded multilingual performance; with only 2000 samples, overfitting risk outweighs gains

## Dataset

| Split | Samples |
|---|---|
| Train | 2,000 (audio + transcriptions) |
| Test | 100 (audio only) |

Language distribution: English (~50%), Hindi (~25%), Tamil (~25%)

## Results

| Metric | Value |
|---|---|
| Validation WER (20-sample subset) | 23.16% |
| Test predictions generated | 100 |

## Pipeline

```
Audio Input (.wav, 16kHz)
        ↓
Feature Extraction (Log-Mel Spectrogram)
        ↓
Language Detection (auto)
        ↓
Whisper Large-V3 Encoder-Decoder (beam search)
        ↓
Transcription in native script
        ↓
submission.csv
```

## How to Run

1. Open `21f3000028-nppe-2.ipynb` on Kaggle
2. Add the competition dataset: `multilingual-speech-recognition`
3. Add your HuggingFace token as a Kaggle secret: `HF_TOKEN`
4. Enable GPU (T4) runtime
5. Run all cells

## Key Design Decision

Whisper Large-V3 already handles English, Hindi, and Tamil strongly zero-shot. Fine-tuning on 2000 samples risked overfitting without meaningful WER gains, and smaller Whisper variants showed significantly worse multilingual performance in experiments.

## License

MIT
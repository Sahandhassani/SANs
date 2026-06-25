# SANS Multi-Stage Transfer Learning Framework

This project implements the first stage of a multi-stage transfer learning framework for predictive risk assessment of Spaceflight-Associated Neuro-Ocular Syndrome (SANS).

## Current Phase

The current version performs domain-specific pretraining on the Kermany/OCT2017 retinal OCT dataset. The model is trained as a binary classifier:

- NORMAL: healthy
- CNV, DME, DRUSEN: unhealthy

This stage is intended to learn general OCT morphology features before later fine-tuning on NASA OSDR rodent and human SANS-related datasets.

## Planned Phases

### Phase 1: Terrestrial OCT pretraining

Train OCT feature extractors on Kermany/OCT2017.

Models:

- Custom CNN
- AlexNet
- ResNet18
- ResNet50
- DenseNet121
- EfficientNet-B0
- ViT-B/16

### Phase 2: OSDR rodent/human fine-tuning

Use NASA OSDR datasets, especially OSD-679 and OSD-680, to adapt the model toward SANS-related ocular and optic nerve morphology.

### Phase 3: Cross-species generative adaptation

Add a generative module to normalize rodent OCT into human-like OCT appearance, or synthesize human SANS-like OCT from rodent SANS analog data.

### Phase 4: Predictive risk assessment

Use pre-exposure structural markers such as BMO geometry, RNFL thickness, ONH displacement, and retinal thickening to predict high-risk ocular profiles before long-duration spaceflight.

## Setup

```bash
pip install -r requirements.txt
```

## Download Kermany dataset

```bash
sbatch jobs/job_download_kermany.sh
```

## Download NASA OSDR data

```bash
sbatch jobs/job_download_osdr.sh
```

## Train models

```bash
sbatch jobs/job_train_alexnet.sh
sbatch jobs/job_train_resnet18.sh
sbatch jobs/job_train_customcnn.sh
```

## Outputs

Checkpoints:

```text
outputs/checkpoints/
```

Training logs:

```text
outputs/logs/
```

Metrics:

```text
outputs/results/
```

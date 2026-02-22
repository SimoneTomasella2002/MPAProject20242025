# Deep Learning Project

Final project for "Predictive Methods for Businesses" at Università di Genova, A.A. 2024/2025

## Authors
- Simone Tomasella (5311626)
- Siria Zuddas (5569720)

## Description
This project aims to recognize 10 different paintings along with their parodies and community-made versions by identifying common visual features. After detecting the painting, it infers the class (e.g., a parody of the Mona Lisa is categorized as Mona Lisa), then uses Ollama running an LLM (DeepSeek R1 1.5B) to generate a description of the original painting and gTTS to produce a spoken explanation.

## Libraries
- [YOLO Ultralytics](https://github.com/ultralytics/ultralytics)
- [pip](https://pypi.org/project/pip/)
- [OpenCV](https://opencv.org/)
- [Ollama](https://github.com/jmorganca/ollama)
- [gTTS](https://pypi.org/project/gTTS/)

## Workflow
1. Detect the painting.
2. Infer which painting it represents.
3. Describe it using the LLM.
4. Convert the text description to speech.

## Environment Setup (venv + dependencies)

### 1) Create the virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
```

Windows (PowerShell):

```powershell
python -m venv .venv
```

### 2) Activate the virtual environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install required libraries

After activation, install all project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4) (Optional) Verify installation

```bash
pip list
```

### 5) Deactivate when finished

```bash
deactivate
```

## Ollama Setup (Required)

This project requires Ollama to be installed and running, otherwise the app cannot generate painting descriptions.

### 1) Install Ollama

Download and install Ollama from:

- https://ollama.com/download

### 2) Download the required model (only first time)

```bash
ollama pull deepseek-r1:1.5b
```

### 3) Start Ollama every time before running this project

```bash
ollama serve
```

Keep this process running while using the app.

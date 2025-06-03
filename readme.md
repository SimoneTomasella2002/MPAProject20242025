# Deep Learning Project

Final project for "Predictive Methods for Businesses" at Università di Genova, A.A. 2024/2025

## Authors
- Simone Tomasella (5311626)
- Siria Zuddas (code unknown)

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
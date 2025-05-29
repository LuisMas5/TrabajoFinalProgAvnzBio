# Pruning y Cuantización con TensorFlow

Este repositorio contiene un ejemplo práctico de reducción de tamaño de redes neuronales aplicando técnicas de **pruning** y **cuantización** usando TensorFlow y la biblioteca `tensorflow-model-optimization`.

## Descripción

Se utiliza una red neuronal convolucional simple entrenada sobre el dataset MNIST para ilustrar cómo estas técnicas permiten disminuir el tamaño del modelo manteniendo una alta precisión.

## Requisitos

- Python 3.10
- TensorFlow 2.13.0 (compatible con macOS)
- TensorFlow Model Optimization Toolkit

## Instalación

```bash
conda create -n tf310 python=3.10
conda activate tf310
pip install tensorflow-macos==2.13.0
pip install tensorflow-model-optimization
pip install -r requirements.txt
```

## Uso

Ejecutar el script principal para entrenar, aplicar pruning y cuantización:

```bash
python modelo.py
```

## Resultados

El script guarda el modelo original, podado y cuantizado, y muestra la comparación de tamaños y precisión.

## Referencias

- Han et al. (2015) [Deep Compression](https://arxiv.org/abs/1510.00149)
- Jacob et al. (2018) [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

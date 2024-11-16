# Simulazione Ambiente con CUDA C

Questo progetto implementa una simulazione di formiche che interagiscono con un ambiente bidimensionale. La computazione è ottimizzata utilizzando CUDA C, mentre la visualizzazione è gestita tramite Python.

## Struttura del progetto

- `main.cu`: Punto d'ingresso del programma.
- `kernel.cu` e `kernel.cuh`: Definizioni e implementazioni delle funzioni CUDA.
- `randomStatesKernel.cu` e `randomStatesKernel.cuh`: Gestione degli stati casuali.
- `visualize.py`: Script Python per la visualizzazione dei risultati.

## Requisiti

1. **CUDA Toolkit** installato sul sistema.
2. **Python** con le seguenti librerie:
   - `numpy`
   - `matplotlib`

## Come Eseguire

1. Compila il progetto CUDA:
   ```bash
   nvcc main.cu kernel.cu randomStatesKernel.cu -o simulazione

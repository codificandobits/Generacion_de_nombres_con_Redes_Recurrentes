# Generación de nombres usando Redes Neuronales Recurrentes

Código fuente de [este](https://youtu.be/aA9QaPu_QpA) video, en donde se muestra cómo usar una simple arquitectura de Red Neuronal Recurrente (RNN) para generar nombres de dinosaurios.

El set de entrenamiento contiene 1536 nombres de dinosaurios. El modelo se implementa en Keras a partir de una celda recurrente (SimpleRNN), y el entrenamiento y generación de nombres se lleva a cabo caracter por caracter.

Tras 10000 iteraciones de entrenamiento se obtienen nombres generados como estos:
- *onorosacasuinis*
- *ceidotis*
- *lidoremhasaus*
- *yphisa*
- *osasovidus*

## Dependencias
Keras==2.2.4
numpy==1.16.3
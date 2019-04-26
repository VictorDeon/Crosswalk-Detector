"""
Arquivo utilizado para detectar faixas de pedestres
a partir de imagens passadas como parâmetro
"""

from cnn import CrosswalkCNN
import cv2 as cv


# Rodar o algoritmo de classificação por imagem
# CrosswalkCNN.test("./test/r5.jpg")

# Rodar o algoritmo de treinamento
# CrosswalkCNN.training()

# Validar algoritmo
CrosswalkCNN.validate()

"""
Arquivo usado para mudar uma quantidade N de arquivos
de uma pasta para outra e renomear os arquivos se precisar
"""

import cv2
import os
import numpy as np
from PIL import Image
import pathlib


def change_imagens(current_folder, destination_folder, name="crosswalk", qtd=0, dim=(128, 64)):
    """
    Arquivo usado para mudar uma quantidade N de arquivos
    de uma pasta para outra e renomear os arquivos se precisar
    """

    img_path = [os.path.join(current_folder, file) for file in os.listdir(current_folder)]
    qtd_img = 1

    for img in img_path:
        img_name = os.path.split(img)[1].split("/")[0]
        extension = os.path.split(img_name)[1].split(".")[0]

        new_name = name
        saved_name = new_name + "_" + str(qtd_img + qtd)
        print(img_name + " -> " + saved_name + ".jpg")

        try:
            saved_folder = destination + "/"

            # carrega a imagem
            img = Image.open(current_folder + "/" + img_name)
            # converte a imagem (PIL) para numpy array
            imgNp = np.array(img,'uint8')
            # redimensionar a imagem
            imgNp = cv2.resize(imgNp, dim)

            # Cria a pasta positivas_final e salva as imagens
            pathlib.Path(saved_folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(saved_folder + saved_name + ".jpg", imgNp)

            qtd_img += 1

        except ValueError:
            print('.')


folder = 'test'
destination = 'new_folder'

change_imagens(folder, destination)

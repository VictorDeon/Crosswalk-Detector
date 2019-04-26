"""
Arquivo utilizado para detectar faixas de pedestres
a partir de imagens passadas como parâmetro
"""

from imutils.video import VideoStream, FPS
import cv2 as cv
import imutils
import numpy as np
import pathlib

pasta_destino = 'dataset/frames/'


def video_run():
    """
    Função para extrair imagens de vídeos
    """

    # Inicializa o vídeo stream, inicia o sensor da câmera e inicializa o FPS counter
    # Para usar a webcam, câmera externa é VideoStream(src=1).start()
    # e vídeo é VideoStream("nome_do_video.mp4").start()
    video = VideoStream('video.mp4').start()

    # Inicia a captura e contabiliza os FPS (frames por segundos)
    fps = FPS().start()

    # Loop sobre os frames do vídeo stream
    while True:
        # Obtém o frame de vídeo stream e redimensiona para ter uma largura máxima de 900 pixels
        frame = video.read()

        try:
            # converte a imagem para numpy array
            imgNp = np.array(frame, 'uint8')
            # redimensionar a imagem
            imgNp = cv.resize(imgNp, (128, 64))

            # Cria a pasta com as imagens e salva as imagens
            pathlib.Path(pasta_destino).mkdir(parents=True, exist_ok=True)
            cv.imwrite(pasta_destino + "crosswalk_" + str(fps._numFrames) + ".jpg", imgNp)

        except ValueError:
            print('.')

        # Mostra a saída
        cv.imshow("Vídeo Stream", frame)
        key = cv.waitKey(1) & 0xFF

        # Se pressionada a tecla 'q' encerra o loop
        if key == ord('q'):
            break

        # Atualiza o FPS counter
        fps.update()

    # Para o timer
    fps.stop()
    print("Tempo total de captura: {:.2f}".format(fps.elapsed()))
    print("FPS: {:.2f}".format(fps.fps()))

    # Limpa tudo
    video.stop()
    cv.destroyAllWindows()


video_run()

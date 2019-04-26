from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from constants import CROSSWALK, NOT_CROSSWALK
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import cv2 as cv
import os


class CrosswalkCNN(object):
    """
    Classe de redes neurais convolucionais para detecção
    de faixas de pedestre pela visão de um pedestre.
    """

    @classmethod
    def training(cls):
        """
        Método reponsável por realizar o treinamento da rede neural
        """

        __classifier = cls.__create_cnn()

        # Cria novas imagens a partir das imagens existentes
        __normalization = 1./255

        # ROTATION_RANGE = Grau de rotação da imagem
        # HORIZONTAL_FLIP = Vai fazer giros horizontais nas imagens
        # SHEAR_RANGE = Faz a mudança dos pixels para outra direção
        # HEIGHT_SHIFT_RANGE = Faixa de mudança da altura da imagem
        # ZOOM_RANGE = Faixa de mudança do zoom da imagem
        __img_generator = ImageDataGenerator(
            rescale=__normalization,
            rotation_range=7,
            horizontal_flip=True,
            shear_range=0.2,
            height_shift_range=0.07,
            zoom_range=0.2
        )

        # Cria novas imagens de teste a partir das imagens existentes
        __test_img_generator = ImageDataGenerator(rescale=__normalization)

        # Criar a base de dados de treinamento
        # TARGET_SIZE = Tamanho das imagens (altura, largura)
        # BATCH_SIZE = De quantas em quantas imagens será feito o treinamento (de 29 em 29 no caso) - 956/32 = 29
        # CLASS_MODE = Tipo de saída, no caso é binario
        __training_base = __img_generator.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 128),
            batch_size=32,
            class_mode='binary',
            classes=['not_crosswalk', 'crosswalk']
        )

        print(__training_base.class_indices)

        # Criar a base de dados de teste
        # BATCH_SIZE = De quantas em quantas imagens será feito o teste (de 10 em 10 no caso) - 320/32 = 10
        __test_base = __test_img_generator.flow_from_directory(
            'dataset/testing_set',
            target_size=(64, 128),
            batch_size=32,
            class_mode='binary',
            classes=['not_crosswalk', 'crosswalk']
        )

        print(__training_base.class_indices)

        # Irá salvar o modelo a cada ciclo (epoch) e armazenar o melhor
        # FILEPATH = Nome do modelo salvo a cada ciclo
        # MONITOR = Atributo que será monitorado
        # VERBOSE = Aparecer na tela feedbacks sobre o checkpoint
        checkpointer = ModelCheckpoint(
            filepath="classifiers/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor="val_loss",
            verbose=1
        )

        # Para o treinamento quando o mesmo não evolui mais
        # MONITOR = Atributo que será monitorado no treinamento
        # PATIENCE = Quantidade de vezes que deve ocorrer a não evolução até parar
        # VERBOSE = Aparecer na tela feedbacks sobre a parada do treinamento
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=2,
            verbose=1
        )

        # Fazer o treinamento
        # STEPS_PER_EPOCH = Quantidade de imagens que será treinada por ciclo, ideal o total de imagens
        # EPOCHS = Quantidade de ciclos que será treinado, quanto maior melhor
        # VALIDATION_DATA = Imagens de teste
        # VALIDATION_STEPS = Quantidade de imagens de teste por ciclo
        __classifier.fit_generator(
            __training_base,
            steps_per_epoch=6654,
            epochs=20,
            validation_data=__test_base,
            validation_steps=1260,
            callbacks=[checkpointer, early_stopping]
        )

    @classmethod
    def validate(cls):
        """
        Faz validação de acuracia e cria a matriz de confusão.
        """

        crosswalk, not_crosswalk = cls.__create_img_numpy_array()

        print("Estamos fazendo os cálculos, por favor espere.")
        crosswalk_success, crosswalk_fail, crosswalk_average = cls.__get_validations(crosswalk)
        not_crosswalk_fail, not_crosswalk_success, not_crosswalk_average = cls.__get_validations(not_crosswalk, True)

        print("###### CROSSWALK #####")
        print("Média: %.2f%% de acerto" % (crosswalk_average * 100))
        print("Sucesso: %02d" % crosswalk_success)
        print("Falha: %02d" % crosswalk_fail)

        print("###### NOT CROSSWALK #####")
        print("Média: %.2f%% de acerto" % (not_crosswalk_average * 100))
        print("Sucesso: %02d" % not_crosswalk_success)
        print("Falha: %02d" % not_crosswalk_fail)

    @classmethod
    def __get_validations(cls, imgs, not_crosswalk=False):
        """
        Faz a validação e retorna a quantidade de
        casos de sucesso, falha e a média de acerto.
        """

        __model = cls.__get_classifier()

        predictions = []
        qtd_crosswalk = 0
        qtd_not_crosswalk = 0
        sum = 0
        for img in imgs:
            __prediction = __model.predict(img)
            predictions.append(__prediction)
            sum += float(__prediction)
            if __prediction >= 0.5:
                qtd_crosswalk += 1
            else:
                qtd_not_crosswalk += 1

        average = sum/len(predictions)

        if not_crosswalk:
            average = abs(average - 1)

        return qtd_crosswalk, qtd_not_crosswalk, average

    @classmethod
    def __create_img_numpy_array(cls):
        """
        Pega todas as imagens do dataset de teste
        e gera um array de imagens numpy
        """

        crosswalk_dataset_path = "./dataset/testing_set/crosswalk"
        not_crosswalk_dataset_path = "./dataset/testing_set/not_crosswalk"
        test_path = "./test"

        crosswalk_path = [os.path.join(crosswalk_dataset_path, file) for file in os.listdir(crosswalk_dataset_path)]
        not_crosswalk_path = [os.path.join(not_crosswalk_dataset_path, file) for file in os.listdir(not_crosswalk_dataset_path)]

        crosswalk_array_list = []
        for path in crosswalk_path:
            img = cls.__img_processing(path)
            crosswalk_array_list.append(img)

        not_crosswalk_array_list = []
        for path in not_crosswalk_path:
            img = cls.__img_processing(path)
            not_crosswalk_array_list.append(img)

        return crosswalk_array_list, not_crosswalk_array_list

    @classmethod
    def test(cls, path):
        """
        Ao inserir uma nova image, classifica-la
        como faixa de pedestre ou não faixa de pedestre.

        new_image = Imagem no formato de matriz numpy.

        return classificação das imagens entre ['not_crosswalk', 'crosswalk']
        """

        # Pega a imagem
        img = cls.__img_processing(path)

        __classifier = cls.__get_classifier()

        __prediction = __classifier.predict(img)

        # 0 < prediction < 0.5 = not_crosswalk
        # 0.5 <= prediction < 1 = crosswalk
        print(__prediction)
        is_crosswalk = False
        if float(__prediction) >= 0.5:
            text = CROSSWALK
            is_crosswalk = True
        else:
            text = NOT_CROSSWALK

        print(is_crosswalk, ": " + text)
        return text, is_crosswalk

    @classmethod
    def __img_processing(cls, img):
        """
        Faz todo o processamento da imagem para
        ser usada no algoritmo.
        """

        # Pega a imagem
        new_image = cv.imread(img)

        # Normalizar a imagem (largura=64, altura=64, canais=3)
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)
        new_image = cv.resize(new_image, (128, 64), interpolation=cv.INTER_AREA)
        new_image = new_image.astype('float32')
        new_image /= 255

        # Formatar para o formato do tensorFlow (qtd, altura, largura, dimensoes)
        # Verificar se isso não vai dar problema.
        new_image = np.expand_dims(new_image, axis = 0)

        return new_image

    @classmethod
    def __get_classifier(cls):
        """
        Pega o modelo treinado de duas formas
        o modelo com estrutura e os melhores pesos e o modelo
        completo com os melhores pesos armazenados no checkpoint.
        """

        # Podemos também pegar o modelo do checkpoint
        __classifier = load_model('classifiers/crosswalk.hdf5')

        return __classifier

    @classmethod
    def __create_cnn(cls):
        """
        Cria a rede neural convolucional
        """

        __classifier = Sequential()

        # Saída é uma matriz de caracteristicas
        # Imagem (h * w * d) = (64 * 128 * 3) = 24.576 neuronios
        # Filtro (fh * fw * fd) = (3 * 3 * 1) = 9 neuronios
        # Matriz de caracteristicas (h - fh + 1) x (w - fw + 1) x 1 = (64 - 3 + 1) x (128 - 3 + 1) = 62 x 126 = 7.812 neuronios
        # PADDING = Adicionar pixels com 0 em volta da imagem para que a cada camada não se perca o tamanho dela.
        __classifier.add(Conv2D(
            filters=32,
            kernel_size=(3, 3), # (height, width)
            input_shape=(64, 128, 3),
            activation='relu',
            padding='same'
        ))

        # Vai acelerar o processamento, ou seja, vai pegar o mapa de caracteristicas
        # e vai normaliza os valores em uma escala entre 0 e 1
        __classifier.add(BatchNormalization())

        # Usado para pegar o maior valor do mapa de caracteristicas, realçando-as.
        # Matriz de caracteristica (h x w x d) = 62 x 128 x 1 = 7.812 neuronios
        # Pooling Matrix (ph x pw x pd) = 2 x 2 x 1 = 4
        # Matriz de características realçadas (h/ph x w/pw x d/pd) = 31 x 64 x 1 = 1.984 neuronios
        __pooling_matrix = (2, 2)  # matriz 2x2
        __classifier.add(MaxPooling2D(pool_size=__pooling_matrix))

        # Segunda camada de convolução
        __classifier.add(Conv2D(
            filters=64,
            kernel_size=(3, 3), # (height, width)
            input_shape=(64, 128, 3),
            activation='relu',
            padding='same'
        ))
        __classifier.add(BatchNormalization())
        __classifier.add(MaxPooling2D(pool_size=__pooling_matrix))

        # Terceira camada de convolução
        __classifier.add(Conv2D(
            filters=128,
            kernel_size=(3, 3), # (height, width)
            input_shape=(64, 128, 3),
            activation='relu',
            padding='same'
        ))
        __classifier.add(BatchNormalization())
        __classifier.add(MaxPooling2D(pool_size=__pooling_matrix))


        # Adicionar o flattening para vetorizar a matriz para entrar na rede neural densa
        __classifier.add(Flatten())

        # Adicionar a primeira camada da rede neural densa (camada de entrada).
        # UNITS = Quantidade de neuronios que farão parte da camada ((neuronios + 1)/2)
        # ACTIVATION = Função de ativação
        # KERNEL_INITIALIZER = Como você vai fazer a inicialização dos pesos
        __neurons = 256
        __classifier.add(Dense(
            units=__neurons,
            activation='relu',
            kernel_initializer='random_uniform'
        ))

        # Vamos zerar 20% dos valores de entrada para evitar o overfitting
        __classifier.add(Dropout(0.2))

        # Segunda camada densa
        __neurons = 256
        __classifier.add(Dense(
            units=__neurons,
            activation='relu',
            kernel_initializer='random_uniform'
        ))
        __classifier.add(Dropout(0.2))

        # Vamos criar a camada de saída da rede neural densa
        __output = 1
        __classifier.add(Dense(
            units=__output,
            activation='sigmoid'
        ))

        # Vamos compilar a rede neural
        __classifier.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return __classifier

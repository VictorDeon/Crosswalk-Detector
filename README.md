# 2019-01 - Bengala - Treinamento para detecção de faixas de pedestre

Treinamento do algoritmo de detecção de faixa de pedestre utilizando redes neurais convolucionais

## Instalação

```
sudo apt-get install python-opencv
sudo apt-get install libhdf5-serial-dev
pip install -r requirements.txt
```

## Uso

Para usar o algoritmo treinado é só pegar o modelo **crosswalk.hdf5** e inserir no seu
projeto de detecção de faixas usando redes neurais.

Exemplo de uso no arquivo **main.py**

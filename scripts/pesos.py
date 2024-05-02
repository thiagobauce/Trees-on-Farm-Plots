import cv2
import os
from tqdm import tqdm

# Definindo as classes
classes = {"background": 0, "talhoes": 0, "arvores": 0}

# Pasta contendo as imagens
pasta_imagens =  r'/home/guatambu/bauce_ds/projeto/dataset/data_am/treino_am/label/512'
print(pasta_imagens)

# Contador de arquivos processados
arquivos_processados = 0

# Contador total de arquivos
total_arquivos = len(os.listdir(pasta_imagens))

for filename in tqdm(os.listdir(pasta_imagens), desc="Processando imagens", unit="imagem", total=total_arquivos):
    if filename.endswith(".png"):
        image_path = os.path.join(pasta_imagens, filename)
        image = cv2.imread(image_path)

        # Convertendo imagem para RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Percorrendo todos os pixels da imagem
        for row in image_rgb:
            for pixel in row:
                if (pixel == [0, 0, 0]).all():  # Preto
                    classes["background"] += 1
                elif (pixel == [255, 0, 0]).all():  # Vermelho
                    classes["talhoes"] += 1
                elif (pixel == [0, 0, 255]).all():  # Azul
                    classes["arvores"] += 1

        # Incrementando o contador de arquivos processados
        arquivos_processados += 1

# Função para calcular o peso de cada classe
def calcular_pesos(total_pixels):
    if total_pixels == 0:
        print("Nenhuma imagem processada ou pixels encontrados.")
        return

    for classe, pixels in classes.items():
        peso = pixels / total_pixels * 100
        print(f"Peso da classe {classe}: {peso:.2f}%")

# Calculando o total de pixels em todas as imagens
total_pixels = sum(classes.values())

# Calculando o peso de cada classe
calcular_pesos(total_pixels)

print(f"Processamento concluído! Total de arquivos processados: {arquivos_processados}/{total_arquivos}.")
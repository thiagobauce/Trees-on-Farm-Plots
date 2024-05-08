import argparse
import os
import time
from mmseg.apis import init_model, inference_model
from PIL import Image
import numpy as np


def main(args):
    model_path = args.model
    img_dir = args.images
    output_path = args.output
    config_path = args.config
    model_type = args.type

    # Inicializa o modelo
    model = init_model(config_path, model_path)

    # Lista todos os arquivos no diretório img_dir
    img_files = os.listdir(img_dir)

    total_files = len(img_files)
    files_processed = 0

    start_time = time.time()

    for img_file in img_files:
        out_name = os.path.splitext(os.path.basename(img_file))[0]
        img = Image.open(os.path.join(img_dir, img_file))
        img_np = np.array(img)
        pred = inference_model(model, img_np)
        result = pred.pred_sem_seg.data.squeeze(0).cpu().numpy()
        palette = [0, 0, 0, 255, 0, 0, 0, 0, 255]  # Palette para visualização
        segmented_img = Image.fromarray(result.astype('uint8'))
        segmented_img.putpalette(palette)

        # Cria o diretório de saída se ele não existir
        os.makedirs(output_path, exist_ok=True)

        output_file = os.path.join(output_path, f"{out_name}_segmented.png")
        segmented_img.save(output_file)

        files_processed += 1
        time_elapsed = time.time() - start_time
        time_per_file = time_elapsed / files_processed if files_processed > 0 else 0
        eta = (total_files - files_processed) * time_per_file if files_processed > 0 else 0

        # Limpa a linha anterior e imprime a nova linha
        print(f"\rProcessed: {files_processed}/{total_files}, ETA: {eta:.2f} seconds, Time Elapsed: {time_elapsed:.2f} seconds", end='')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on mmseg')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--images', type=str, help='Path to the directory containing RGB test images')
    parser.add_argument('--output', type=str, help='Path to the output directory')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--type', type=str, default='segmentation', help='Type of model (e.g., segmentation)')
    main(parser.parse_args())

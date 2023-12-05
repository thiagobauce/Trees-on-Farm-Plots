import os
import numpy as np
import cv2
import rasterio
import geopandas
from PIL import Image
import argparse

def main(args):
    
    diretorio_raiz=args.directory

    classes = {
        "mascara": 0,
        "talhoes": 1,
        "arvores": 2
    }

    diretorios = [os.path.join(diretorio_raiz, nome) 
                    for nome in os.listdir(diretorio_raiz) 
                        if os.path.isdir(os.path.join(diretorio_raiz, nome))
                    ]

    print(diretorios)

    for diretorio in diretorios:
        # Listar os arquivos no diretório atual
        arquivos = os.listdir(diretorio)

        # Inicializar variáveis para armazenar os caminhos dos arquivos
        path_shp_talhoes = ''
        path_shp_arvores = ''
        path_shp_mascara = ''
        path_tif = ''

        for arquivo in arquivos:
            if arquivo.endswith('.shp'):
                nome_camada, _ = os.path.splitext(arquivo)
                partes = nome_camada.split('_')

                if 'talhoes' in partes:
                    path_shp_talhoes = os.path.join(diretorio, arquivo)
                    print(path_shp_talhoes)
                elif 'arvores' in partes:
                    path_shp_arvores = os.path.join(diretorio, arquivo)
                    print(path_shp_arvores)
                elif 'mascara' in partes:
                    path_shp_mascara = os.path.join(diretorio, arquivo)
                    print(path_shp_mascara)

            elif arquivo.endswith('.tif'):
                path_tif = os.path.join(diretorio, arquivo)

        ortofoto = read_file_tif(path_tif)

        width = ortofoto.width
        height = ortofoto.height
        n = ortofoto.count
        func_latlon_xy = ortofoto.index

        #print('Largura e Altura: ', (width, height))
        #print('Num de canais: ', n)

        talhoes = read_file_shp(path_shp_talhoes,ortofoto)
        arvores = read_file_shp(path_shp_arvores,ortofoto)
        mascara = read_file_shp(path_shp_mascara,ortofoto)

        output_seg_dir = os.path.join(diretorio_raiz, "seg_dir")
        os.makedirs(output_seg_dir, exist_ok=True)

        path_out_seg_label = os.path.join(diretorio_raiz, "seg_dir/label")
        path_out_seg_rgb = os.path.join(diretorio_raiz, "seg_dir/rgb")

        patch_size = args.patch
        step = args.step

        for size in patch_size:
            os.makedirs(os.path.join(output_seg_dir, f"label/{size}"), exist_ok=True)
            os.makedirs(os.path.join(output_seg_dir, f"rgb/{size}"), exist_ok=True)


        r = ortofoto.read(1)
        g = ortofoto.read(2)
        b = ortofoto.read(3)

        label_mask = get_mask(width,height, func_latlon_xy,mascara,classes)
        label_tree = get_tree(width,height, func_latlon_xy,arvores,classes)
        label_talhoes = get_talhoes(width,height, func_latlon_xy,talhoes,classes)

        for p, s in zip(patch_size, step):
            print(p,s)
            crop_imgs(width,height,path_out_seg_label, path_out_seg_rgb,
                    r,g,b,p,s,label_tree,label_mask,[0, 0, 0, 255, 0, 0])

            crop_imgs(width,height,path_out_seg_label, path_out_seg_rgb,
                    r,g,b,p,s,label_talhoes,label_mask,[0, 0, 255, 0, 0, 0])

#this function read a tif file (ortofoto)
def read_file_tif(path_tif):
    return rasterio.open(path_tif)

#read a shapefile like layer, fields and trees
def read_file_shp(path_shp,orto):
    shp = geopandas.read_file(path_shp)
    return shp.to_crs(orto.crs)


#get the tress coords
def get_tree(width,height,func_latlon_xy,shp,classes):
    label_tree = np.zeros((height, width), dtype=np.uint8)

    for idx, geometry in enumerate(shp.geometry):
        if geometry is None:
            continue

        if geometry.geom_type == 'Polygon':
            polygons = [geometry] 
        elif geometry.geom_type == 'MultiPolygon':
            polygons = geometry.geoms

        for polygon in polygons:
            points_latlon = polygon.exterior.coords[:]
            points_xy = [func_latlon_xy(point[0], point[1])
                         for point in points_latlon]
            points_xy = np.array(points_xy, np.int32)[:, ::-1]

            class_name = "arvores"
            class_value = classes[class_name]

            cv2.fillPoly(label_tree, [points_xy], class_value)

    Image.fromarray(label_tree*255)

    return label_tree

#get the masks coords
def get_mask(width,height,func_latlon_xy,shp,classes):
    label_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, geometry in enumerate(shp.geometry):
        if geometry is None:
            continue

        if geometry.geom_type == 'Polygon':
            polygons = [geometry] 
        elif geometry.geom_type == 'MultiPolygon':
            polygons = geometry.geoms

        for polygon in polygons:
            points_latlon = polygon.exterior.coords[:]
            points_xy = [func_latlon_xy(point[0], point[1])
                         for point in points_latlon]
            points_xy = np.array(points_xy, np.int32)[:, ::-1]

            class_name = "mascara"
            class_value = classes[class_name]
            
            cv2.fillPoly(label_mask, [points_xy], class_value)

    Image.fromarray(label_mask*255)

    return label_mask

#get the fieldss coords
def get_talhoes(width, height, func_latlon_xy, shp,classes):
    label_talhoes = np.zeros((height, width), dtype=np.uint8)

    for idx, geometry in enumerate(shp.geometry):
        if geometry is None: #existe algum shape
            continue
        if geometry.geom_type == 'Polygon':
            polygons = [geometry] 
        elif geometry.geom_type == 'MultiPolygon':
            polygons = geometry.geoms

        for polygon in polygons:
            points_latlon = polygon.exterior.coords[:]
            points_xy = [func_latlon_xy(point[0], point[1])
                         for point in points_latlon]
            points_xy = np.array(points_xy, np.int32)[:, ::-1]

            class_name = "talhoes"
            class_value = classes[class_name]

            cv2.fillPoly(label_talhoes, [points_xy], class_value)

    Image.fromarray(label_talhoes * 255)

    return label_talhoes

#this function receive a width, height and rgb from tif file, the rgbs and labels paths, the size of crops ans steps to
#iou 
def crop_imgs(width,height, path_label, path_rgb, r,g,b,patch_size, step, label_tree,label_mask,label_talhoes, diretorio):
    for x in range(0, height - patch_size, step):
        for y in range(0, width - patch_size, step):

            patch_mask = label_mask[x:x+patch_size, y:y+patch_size]
            
            if np.sum(patch_mask) == 0:
              continue
        
            patch_r = r[x:x+patch_size, y:y+patch_size]
            patch_g = g[x:x+patch_size, y:y+patch_size]
            patch_b = b[x:x+patch_size, y:y+patch_size]
        
            patch_label = label_tree[x:x+patch_size, y:y+patch_size] + label_talhoes[x:x+patch_size, y:y+patch_size]
        
            patch_rgb = np.dstack([patch_r, patch_g, patch_b])
            patch_rgb[patch_mask == 0] = [0,0,0]

            nome_camada, _ = os.path.splitext(diretorio)
            partes = nome_camada.split('\\')

            filename_rgb = path_rgb + f'/{patch_size}/{partes[-1]}_patch_{x}_{y}.jpg'
            filename_label = path_label + f'/{patch_size}/{partes[-1]}_patch_{x}_{y}.png'
  
            #more than 1 pixel in img patch
            if np.sum(patch_label) > 0:
                Image.fromarray(patch_rgb).save(filename_rgb)
                img_label = Image.fromarray(patch_label)
                img_label.putpalette([0,0,0, 255,0,0, 0,0,255])
                img_label.save(filename_label)


#TODO call main with argparse -> dir, patch_sizes, step, > 1 pixel 

#IOU score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dataset Generator from tif files")
    parser.add_argument("patch", default=[256, 512, 1024, 2048, 4096], help="array of patches sizes to cut tif file")
    parser.add_argument("step", default=[128, 256, 512, 1024, 2048], help="array of overlay on patches")
    parser.add_argument("dir", default="/home/bauce/projeto/dataset/Arvores/Treino", help="directory of tif and shape files")
    main(parser.parse_args())
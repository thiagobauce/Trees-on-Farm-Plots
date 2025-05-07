"""
The files are dispostos in directories inside others train and test.
Each shape is named following the standard:
*dirname_talhoes.shp
*dirname_trees.shp
*dirname_mask.shp
*dirname.tif

Following this standard we could split easily to read the shapefiles.

Os arquivos estÃ£o dispostos em diretÃ³rios dentro de treino e teste.
Cada camada da ortofoto estÃ¡ nomeada da seguinte maneira:
*nome_da_pasta_talhoes.shp
*nome_da_pasta_arvores.shp
*nome_da_pasta_mask.shp
*nome_da_pasta.tif

Conseguimos facilitar o split dos nomes dos arquivos obedecendo esse padrÃ£o, para ler
os shapefiles.
"""

import os
import numpy as np
import cv2
import rasterio
import geopandas
from PIL import Image
import argparse
import sys


def main(args):
    config_module = __import__(args.config.replace('.py', ''))
    cfg = config_module.config

    diretorio_raiz=cfg.dir

    orto_paths = []

    for file in os.listdir(diretorio_raiz):
        if file.endswith('.tif'):
            orto_paths.append(file)
    
    #print(orto_paths,'\n',shp_paths,'\n',talhoes_path)
    #input()

    print("---------------------------- Iniciando processamento ---------------------------- ")
    print(f"Total de ortofotos a serem processadas: {len(orto_paths)}\n")
    print(orto_paths)
    for o, orto_path in enumerate(orto_paths):
        print(f'Processando ortofoto: {(o+1)}/{len(orto_paths)}: {orto_path}')
        filename_orto = os.path.basename(orto_path)[:-4]
    
        field_path = os.path.join(diretorio_raiz, f"{filename_orto}_talhoes.shp")
        mask_path = os.path.join(diretorio_raiz, f"{filename_orto}_mascara.shp")
        path_tif = os.path.join(diretorio_raiz, orto_path)

        ortofoto = read_file_tif(path_tif)
        width = ortofoto.width
        height = ortofoto.height
        n = ortofoto.count
        func_latlon_xy = ortofoto.index


        talhoes = read_file_shp(field_path,ortofoto)
        mascara = read_file_shp(mask_path,ortofoto)
        
        output_dataset_dir = os.path.join(diretorio_raiz, cfg.outputdir)
        path_out_dataset_label = os.path.join(diretorio_raiz, f'{cfg.outputdir}/label')
        path_out_dataset_rgb = os.path.join(diretorio_raiz, f'{cfg.outputdir}/rgb')
        
        #print(output_dataset_dir,path_out_dataset_label,path_out_dataset_rgb)
        
        patch_size = cfg.patch_size
        step = cfg.step
        pallete = cfg.pallete
        
        for size in patch_size:
            os.makedirs(os.path.join(output_dataset_dir, f"label/{size}"), exist_ok=True)
            os.makedirs(os.path.join(output_dataset_dir, f"rgb/{size}"), exist_ok=True)
        
        r = ortofoto.read(1)
        g = ortofoto.read(2)
        b = ortofoto.read(3)
        label_mask = get_label(width,height, func_latlon_xy,mascara)
        label_talhoes = get_label(width,height, func_latlon_xy,talhoes)

        for p, s in zip(patch_size, step):
            print(p,s)
            crop_imgs(width,height,path_out_dataset_label, path_out_dataset_rgb,
                    r,g,b,p,s,label_mask,label_talhoes, filename_orto,pallete)


#this function read a tif file (ortofoto)
def read_file_tif(path_tif):
    return rasterio.open(path_tif)

#read a shapefile like layer, fields and trees
def read_file_shp(path_shp,orto):
    shp = geopandas.read_file(path_shp)
    return shp.to_crs(orto.crs)

#get the masks coords
def get_label(width,height,func_latlon_xy,shp):
    label = np.zeros((height, width), dtype=np.uint8)

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
            cv2.fillPoly(label, [points_xy], 1)

    Image.fromarray(label*255)

    return label

#this function receive a width, height and rgb from tif file, 
#the rgbs and labels paths, the size of crops ans steps to
#iou 
def crop_imgs(width,height, path_label, path_rgb, r,g,b,patch_size, step, 
            label_mask,label_talhoes, filename_orto,pallete):#, 
    
    tam_nparray = patch_size*patch_size
    print(tam_nparray)

    for x in range(0, height - patch_size, step):
        for y in range(0, width - patch_size, step):

            patch_mask = label_mask[x:x+patch_size, y:y+patch_size]
            
            if np.sum(patch_mask) == 0:
              continue
        
            patch_r = r[x:x+patch_size, y:y+patch_size]
            patch_g = g[x:x+patch_size, y:y+patch_size]
            patch_b = b[x:x+patch_size, y:y+patch_size]
        
            patch_label = label_talhoes[x:x+patch_size, y:y+patch_size]
        
            patch_rgb = np.dstack([patch_r, patch_g, patch_b])
            patch_rgb[patch_mask == 0] = [0,0,0]

            filename_rgb = path_rgb + f'/{patch_size}/{filename_orto}_patch_{x}_{y}.jpg'
            #print(filename_rgb)
            filename_label = path_label + f'/{patch_size}/{filename_orto}_patch_{x}_{y}.png'
            #print(filename_label)

            fore = np.sum(patch_label)
            
            #if(2 in patch_label): #theres more than 1 class (trees and talhoes)
            #if((fore > 0)): 
            Image.fromarray(patch_rgb).save(filename_rgb)
            img_label = Image.fromarray(patch_label)
            img_label.putpalette(pallete)
            img_label.save(filename_label)
            #else: #just one or no classes on patch
            #if ((fore > 0) and (fore < (tam_nparray-1))): #more than 1 pixel + and 1 - in img patch
            #    Image.fromarray(patch_rgb).save(filename_rgb)
            #    img_label = Image.fromarray(patch_label)
            #    img_label.putpalette(pallete)
            #    img_label.save(filename_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dataset Generator from tif files")
    parser.add_argument("--config", help="py config file")
    main(parser.parse_args())
    #main()

"""

The files are dispostos in directories inside others train and test.
Each shape is named following the standard:
*dirname_talhoes.shp
*dirname_trees.shp
*dirname_mask.shp
*dirname.tif

Following this standard we could split easily to read the shapefiles.

Os arquivos estão dispostos em diretórios dentro de treino e teste.
Cada camada da ortofoto está nomeada da seguinte maneira:
*nome_da_pasta_talhoes.shp
*nome_da_pasta_arvores.shp
*nome_da_pasta_mask.shp
*nome_da_pasta.tif

Conseguimos facilitar o split dos nomes dos arquivos obedecendo esse padrão, para ler
os shapefiles.
"""

import os
import numpy as np
import cv2
import rasterio
import geopandas
from PIL import Image

def main():

    diretorio_raiz=r'Q:/Arvores/Treino'

    diretorios = [os.path.join(diretorio_raiz, nome) 
                    for nome in os.listdir(diretorio_raiz) 
                        if os.path.isdir(os.path.join(diretorio_raiz, nome))
                    ]
    #TREINO
    #diretorios = ['Q:/Arvores/Treino\\0085', 'Q:/Arvores/Treino\\0229', 'Q:/Arvores/Treino\\0651', 
                  #'Q:/Arvores/Treino\\0680', 'Q:/Arvores/Treino\\0746', 'Q:/Arvores/Treino\\0772', 
                  #'Q:/Arvores/Treino\\0781-13-14-15', 'Q:/Arvores/Treino\\0781-TH-6-7-8-9-10', 'Q:/Arvores/Treino\\0791', 
                  #'Q:/Arvores/Treino\\10101 - ARUEIRA', 'Q:/Arvores/Treino\\10605 - CUBATAO', 
                  #'Q:/Arvores/Treino\\10613 - SANTO ANTONIO DA LIBERDADE', gd
                  #'Q:/Arvores/Treino\\1133', 
                  #'Q:/Arvores/Treino\\2014-2015', 'Q:/Arvores/Treino\\30443 - SANTA RITA', 'Q:/Arvores/Treino\\30452 - SÃO JOSÉ', 
                  #'Q:/Arvores/Treino\\30768 - CASA DA PEDRA', 'Q:/Arvores/Treino\\30774 - N SRA APARECIDA', 
                  #'Q:/Arvores/Treino\\30812 - SÃO SEBASTIÃO', 'Q:/Arvores/Treino\\30990 - BARRINHA', 
                  #'Q:/Arvores/Treino\\40228 - SANTA HELENA', 'Q:/Arvores/Treino\\40491 - FUNDÃO', 
                  #'Q:/Arvores/Treino\\40520 - MALHADOURO', 'Q:/Arvores/Treino\\50010 - FLORESTA', 
                  #'Q:/Arvores/Treino\\50124 - SANTA MARINA', gd 
                  #'Q:/Arvores/Treino\\50141 - GLORIA', gd
                  #'Q:/Arvores/Treino\\50442 - MGA', 'Q:/Arvores/Treino\\50448 - PROGRESSO', 
                  #'Q:/Arvores/Treino\\50581 - MOLECADA', 'Q:/Arvores/Treino\\50710 - BOA ESPERANÇA']
    #TESTE
    #diretorios = ['Q:/Arvores/Teste\\0151', 'Q:/Arvores/Teste\\0221', 'Q:/Arvores/Teste\\0738', 
                  #'Q:/Arvores/Teste\\2048', 'Q:/Arvores/Teste\\30219 - SÃO JORGE', 
                  #'Q:/Arvores/Teste\\50141 - GLORIA', 
                  #'Q:/Arvores/Teste\\50696 - SANTO ANTONIO CAMINHANTE']
    
    #VALIDACAO
    #diretorios = #['Q:/Arvores/Validacao\\0078', 'Q:/Arvores/Validacao\\0151', 'Q:/Arvores/Validacao\\0659', 
                  #'Q:/Arvores/Validacao\\10008 - EROSÃO', 
                  #'Q:/Arvores/Validacao\\1148', 
                  #'Q:/Arvores/Validacao\\30438 - AROEIRAS', 
                  #'Q:/Arvores/Validacao\\50191 - GIRASSOL']

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

        output_arvores_dir = os.path.join(diretorio_raiz, "arvores")
        output_talhoes_dir = os.path.join(diretorio_raiz, "talhoes")
        os.makedirs(output_arvores_dir, exist_ok=True)
        os.makedirs(output_talhoes_dir, exist_ok=True)

        path_out_tree_label = os.path.join(diretorio_raiz, "arvores/arvores_label")
        path_out_tree_rgb = os.path.join(diretorio_raiz, "arvores/arvores_rgb")
        path_out_talhoes_label = os.path.join(diretorio_raiz, "talhoes/talhoes_label")
        path_out_talhoes_rgb = os.path.join(diretorio_raiz, "talhoes/talhoes_rgb")

        patch_size = [256, 512, 1024, 2048, 4096]
        step = [128, 256, 512, 1024, 2048]

        for size in patch_size:
            os.makedirs(os.path.join(output_arvores_dir, f"arvores_rgb/{size}"), exist_ok=True)
            os.makedirs(os.path.join(output_arvores_dir, f"arvores_label/{size}"), exist_ok=True)

            os.makedirs(os.path.join(output_talhoes_dir, f"talhoes_rgb/{size}"), exist_ok=True)
            os.makedirs(os.path.join(output_talhoes_dir, f"talhoes_label/{size}"), exist_ok=True)


        r = ortofoto.read(1)
        g = ortofoto.read(2)
        b = ortofoto.read(3)

        label_mask = get_mask(width,height, func_latlon_xy,mascara)
        label_tree = get_tree(width,height, func_latlon_xy,arvores)
        label_talhoes = get_talhoes(width,height, func_latlon_xy,talhoes)

        for p, s in zip(patch_size, step):
            print(p,s)
            crop_imgs(width,height,path_out_tree_label, path_out_tree_rgb,
                    r,g,b,p,s,label_tree,label_mask)

            crop_imgs(width,height,path_out_talhoes_label, path_out_talhoes_rgb,
                    r,g,b,p,s,label_talhoes,label_mask)

            #crop_imgs(width,height,path_out_tree_rel_label, path_out_tree_rel_rgb,
            #        r,g,b,p,s,label_tree,label_talhoes)


#this function read a tif file (ortofoto)
def read_file_tif(path_tif):
    return rasterio.open(path_tif)

#read a shapefile like layer, fields and trees
def read_file_shp(path_shp,orto):
    shp = geopandas.read_file(path_shp)
    return shp.to_crs(orto.crs)

#get the tress coords
def get_tree(width,height,func_latlon_xy,shp):
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
            cv2.fillPoly(label_tree, [points_xy], 1)

    Image.fromarray(label_tree*255)

    return label_tree

#get the masks coords
def get_mask(width,height,func_latlon_xy,shp):
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
            cv2.fillPoly(label_mask, [points_xy], 1)

    Image.fromarray(label_mask*255)

    return label_mask

#get the fieldss coords
def get_talhoes(width, height, func_latlon_xy, shp):
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
            cv2.fillPoly(label_talhoes, [points_xy], 1)

    Image.fromarray(label_talhoes * 255)

    return label_talhoes

#this function receive a width, height and rgb from tif file, the rgbs and labels paths, the size of crops ans steps to
#iou 
def crop_imgs(width,height, path_label, path_rgb, r,g,b,patch_size, step, label, label_mask):
    for x in range(0, height - patch_size, step):
        for y in range(0, width - patch_size, step):

            patch_mask = label_mask[x:x+patch_size, y:y+patch_size]
            
            if np.sum(patch_mask) == 0:
              continue
        
            patch_r = r[x:x+patch_size, y:y+patch_size]
            patch_g = g[x:x+patch_size, y:y+patch_size]
            patch_b = b[x:x+patch_size, y:y+patch_size]
        
            patch_label = label[x:x+patch_size, y:y+patch_size]
        
            patch_rgb = np.dstack([patch_r, patch_g, patch_b])
            patch_rgb[patch_mask == 0] = [0,0,0]
        
            filename_rgb = path_rgb + f'/{patch_size}/patch_{x}_{y}.jpg'
            filename_label = path_label + f'/{patch_size}/patch_{x}_{y}.png'
        
            #Image.fromarray(patch_rgb).save(filename_rgb)
            #img_label = Image.fromarray(patch_label)
            #img_label.putpalette([0,0,0, 255,0,0])
            #img_label.save(filename_label)

            #more than 1 pixel in img patch
            if np.sum(patch_label) > 0:
                Image.fromarray(patch_rgb).save(filename_rgb)
                img_label = Image.fromarray(patch_label)
                img_label.putpalette([0, 0, 0, 255, 0, 0])
                img_label.save(filename_label)


main()
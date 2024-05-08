from easydict import EasyDict as edict

config = edict()

config.patch_size = [256,512,1024]
config.step = [128,256,512]
config.outputdir = 'data' #treino, teste ou validacao
config.dir = r'/home/guatambu/bauce_ds/projeto/dataset/Arvores/Treino'
config.pallete = [0,0,0, 255,0,0, 0,0,255]

# Este script tem o objetivo de ler imagens de exames oftalmológicos e extrair os resultados para análise

import easyocr as ocr
import pandas as pd
import numpy as np
import time
import os

from PIL import Image, ImageChops, ImageFilter
#from joblib import Parallel, delayed
import re
import csv

ground_truth = {}

with open('ground_truth.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        chave = row['variavel'].strip()
        valor = row['valor'].strip()
        ground_truth[chave] = valor


OCR_CUTOFF = 0.6 # OCR cutoff to include result
IMG_MAG = 4 # Image magnification  
BIN_THRESHOLD = 200 # binarization threshold
IMG_BLUR = 1
VERBOSE = False
save_step = 500
OUTPUT_DIR = "output"
IMAGES_DIR = "imagens"
RECORTES_DIR = 'recortes'
dictionary_list_p1 = []
dictionary_list_p2 = []
row_data = {}

#binarization function
def binarize(img, thresh=200):  
  #convert image to greyscale
  img=img.convert('L') 
  width,height=img.size
  #traverse through pixels 
  for x in range(width):
    for y in range(height):
      #if intensity less than threshold, assign white
      if img.getpixel((x,y)) < thresh:
        img.putpixel((x,y),0)
      #if intensity greater than threshold, assign black 
      else:
        img.putpixel((x,y),255)
  return img

def trim_img(im):
  bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
  diff = ImageChops.difference(im, bg)
  diff = ImageChops.add(diff, diff, 2.0, -100)
  bbox = diff.getbbox()
  if bbox:
    return im.crop(bbox)

def resize_img(img, magnification, binarize_thresh=-1):
  width, height = img.size
  im = img.resize((width*magnification,height*magnification), Image.Resampling.LANCZOS)
  if binarize_thresh != -1:
    im = binarize(im, thresh=binarize_thresh)
  return im


def info_crops(padrao):
  try:
    if padrao == 1:
      return {
          'Patient': [46, 5, 140, 16, ''],
          'DOB_age': [60, 18, 140, 16, '01234567890 ()/'],
          'Algorithm_Ver': [707, 18, 140, 16, ''],
          'Exam_date_OD': [121, 563, 84, 16, '/01234567890/'],
          'Exam_date_OS': [752, 563, 84, 16, '/01234567890/'],
          'Gender': [361, 31, 60, 16, 'MF'],
          'Eye': [16, 60, 32, 70, 'OSD'],
          'Eye_2': [890, 64, 32, 70, 'OSD'],
          'MRR': [161, 421, 46, 29, ''],
      }
    else:
      return {
          'Patient': [45, 4, 140, 16, ''],
          'DOB_age': [58, 19, 140, 16, '01234567890 ()/'],
          'Algorithm_Ver': [390, 19, 140, 16, ''],
          'Exam_date': [694, 18, 140, 16, '01234567890 /'],
          'Gender': [362, 32, 60, 16, 'MF'],
          'Eye': [6, 65, 32, 70, 'OSD'],
          'MRR': [173, 304, 46, 29, ''],
      }
  except Exception as e:
    pass


def exam_crops(padrao):
  try:
    if padrao == 1:
      return {
          'SSI_OD': [250, 563, 60, 16, '.0123456789'],
          'SSI_OS': [880, 563, 60, 16, '.0123456789'],
          'Net_Power_OD': [53, 488, 41, 18, '.0123456789'],
          'Anterior_Power_OD': [107, 489, 41, 18, '.0123456789'],
          'Posterior_Power_OD': [158, 491, 41, 18, '-0123456789.'],
          'Anterior_R_OD': [61, 533, 33, 18, '.0123456789'],
          'Posterior_R_OD': [159, 532, 33, 18, '.0123456789'],
          'Pachy_SNIT_Pachmetry_OD': [286, 461, 38, 18, '-0123456789.'],
          'Pachy_SNIT_Pachmetry_OS': [326, 462, 38, 18, '-0123456789.'],
          'Pachy_SI_Pachmetry_OD': [426, 461, 38, 18, '-0123456789.'],
          'Pachy_SI_Pachmetry_OS': [466, 461, 38, 18, '-0123456789.'],
          'Pachy_Min_Pachmetry_OD': [286, 489, 38, 18, '.0123456789'],
          'Pachy_Min_Pachmetry_OS': [326, 488, 38, 18, '.0123456789'],
          'Pachy_Y_Pachmetry_OD': [426, 488, 38, 18, '-0123456789.'],
          'Pachy_Y_Pachmetry_OS': [466, 488, 38, 18, '-0123456789.'],
          'Pachy_MinMedian_Pachmetry_OD': [286, 515, 38, 18, '-0123456789.'],
          'Pachy_MinMedian_Pachmetry_OS': [326, 515, 38, 18, '-0123456789.'],
          'Pachy_MinMax_Pachmetry_OD': [427, 515, 38, 18, '-0123456789.'],
          'Pachy_MinMax_Pachmetry_OS': [466, 515, 38, 18, '-0123456789.'],
          'Net_Power_OS': [802, 490, 41, 18, '.0123456789'],
          'Anterior_Power_OS': [856, 488, 41, 18, '.0123456789'],
          'Posterior_Power_OS': [908, 491, 41, 18, '-0123456789.'],
          'Anterior_R_OS': [804, 533, 33, 18, '.0123456789'],
          'Posterior_R_OS': [902, 534, 33, 18, '.0123456789'],
          'Epi_Superior_Epithelium_OD': [563, 461, 38, 18, '.0123456789'],
          'Epi_Superior_Epithelium_OS': [596, 460, 38, 18, '.0123456789'],
          'Epi_Inferior_Epithelium_OD': [677, 461, 38, 18, '.0123456789'],
          'Epi_Inferior_Epithelium_OS': [710, 461, 38, 18, '.0123456789'],
          'Epi_Min_Epithelium_OD': [563, 487, 38, 18, '.0123456789'],
          'Epi_Min_Epithelium_OS': [596, 487, 38, 18, '.0123456789'],
          'Epi_Max_Epithelium_OD': [678, 487, 38, 18, '.0123456789'],
          'Epi_Max_Epithelium_OS': [710, 487, 38, 18, '.0123456789'],
          'Epi_StdDev_Epithelium_OD': [564, 513, 38, 18, '.0123456789'],
          'Epi_StdDev_Epithelium_OS': [596, 513, 38, 18, '.0123456789'],  
          'Epi_MinMax_Epithelium_OD': [677, 515, 30, 18, '-0123456789.'],
          'Epi_MinMax_Epithelium_OS': [711, 516, 30, 18, '-0123456789.'],
   
      }
    else:
      return {
          'SSI': [491, 52, 60, 16, '.0123456789'],
          'Net_Power': [76, 386, 41, 18, '.0123456789'],
          'Anterior_Power': [134, 385, 41, 18, '.0123456789.'],
          'Posterior_Power': [191, 387, 41, 18, '-0123456789.'],
          'Anterior_R': [83, 474, 33, 18, '.0123456789'],
          'Posterior_R': [196, 472, 33, 18, '.0123456789'],
          'Pachy_SNIT': [100, 570, 38, 18, '-0123456789.'],
          'Pachy_SI': [212, 570, 38, 18, '-0123456789.'],
          'Pachy_Min': [100, 598, 38, 18, '.0123456789'],
          'Pachy_Y': [210, 598, 38, 18, '-0123456789.'],
          'Pachy_MinMedian': [99, 624, 38, 18, '-0123456789.'],
          'Pachy_MinMax': [210, 626, 38, 18, '-0123456789.'],
          'Epi_Superior': [98, 730, 38, 18, '.0123456789'],
          'Epi_Inferior': [210, 730, 38, 18, '.0123456789'],
          'Epi_Min': [99, 758, 38, 18, '.0123456789'],
          'Epi_Max': [212, 757, 38, 18, '.0123456789'],
          'Epi_StdDev': [99, 785, 38, 18, '.0123456789'],
          'Epi_MinMax': [210, 784, 30, 18, '-0123456789.'],
      }
  except Exception as e:
    pass


#======================Variáveis de referência para a leitura das imagens padrão 1: 4 imagens das cornias
L = 45 # Largura
A = 15 # Altura

#Mapeamento dos pontos para obtencao dos valores das imagens das cornias: [esquerda, topo]
pachymetry_od_crops = { 
  'CO': [270,230],
  'S1': [268,146],
  'S2': [268,98],
  'SN1': [328,172],
  'SN2': [362,134],
  'N1': [354,230],
  'N2': [403,231],
  'IN1': [328,289],
  'IN2': [364,325],
  'I1': [270,314],
  'I2': [269,362],
  'IT1': [211,290],
  'IT2': [173,324],
  'T1': [184,230],
  'T2': [141,230],
  'ST1': [210,171],  
  'ST2': [176,138]
}

pachymetry_os_crops = { 
  'CO': [650,231],
  'S1': [650,147],
  'S2': [651,98],
  'ST1': [711,171],
  'ST2': [744,136],
  'T1': [732,230],
  'T2': [787,232],
  'IT1': [710,291],
  'IT2': [745,325],
  'I1': [656,319],
  'I2': [652,363],
  'IN1': [591,289],
  'IN2': [558,323],
  'N1': [567,230],
  'N2': [520,232],
  'SN1': [592,170],
  'SN2': [559,138]
}

epithelium_od_crops = { 
  'CO': [275,741],
  'S1': [272,656],
  'S2': [271,610],
  'SN1': [331,681],
  'SN2': [366,646],
  'N1': [356,740],
  'N2': [405,741],
  'IN1': [332,800],
  'IN2': [368,835],
  'I1': [272,822],
  'I2': [273,873],
  'IT1': [214,800],
  'IT2': [178,835],
  'T1': [189,742],
  'T2': [140,741],
  'ST1': [214,680],  
  'ST2': [182,648]
}

epithelium_os_crops = { 
 'CO': [656,742],
  'S1': [654,654],
  'S2': [655,610],
  'ST1': [715,681],
  'ST2': [748,648],
  'T1': [740,740],
  'T2': [788,741],
  'IT1': [715,800],
  'IT2': [748,834],
  'I1': [655,825],
  'I2': [654,874],
  'IN1': [594,800],
  'IN2': [560,834],
  'N1': [572,742],
  'N2': [524,744],
  'SN1': [596,681],
  'SN2': [564,649]
}

# Padrão 2
pachymetry_os_cropsp2 = { 
  'CO': [450,612],
  'S1': [451,538],
  'S2': [451,496],
  'ST1': [504,558],
  'ST2': [531,532],
  'T1': [528,612],
  'T2': [571,612],
  'IT1': [508,664],
  'IT2': [538,696],
  'I1': [456,685],
  'I2': [454,727],
  'IN1': [403,664],
  'IN2': [374,694],
  'N1': [379,612],
  'N2': [339,611],
  'SN1': [404,558],
  'SN2': [375,534]
}

epithelium_os_cropsp2 = { 
  'CO': [738,610],
  'S1': [736,537],
  'S2': [734,499],
  'ST1': [788,560],
  'ST2': [821,530],
  'T1': [812,613],
  'T2': [854,612],
  'IT1': [790,664],
  'IT2': [822,695],
  'I1': [738,686],
  'I2': [740,728],
  'IN1': [684,661],
  'IN2': [658,693],
  'N1': [663,612],
  'N2': [625,611],
  'SN1': [687,558],
  'SN2': [660,532]
}

lista = ['S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']


def get_map_crops(padrao):
  try:
    if padrao == 1:
      map_crops = {
        'CO_POD': [pachymetry_od_crops['CO'][0], pachymetry_od_crops['CO'][1], L, A,'0123456789'],
        'CO_POS': [pachymetry_os_crops['CO'][0], pachymetry_od_crops['CO'][1], L, A,'0123456789'],
        'CO_EOD': [epithelium_od_crops['CO'][0], epithelium_od_crops['CO'][1], L, A,'0123456789'],
        'CO_EOS': [epithelium_os_crops['CO'][0], epithelium_os_crops['CO'][1], L, A,'0123456789']
      }
      for l in lista:
        for i in range(1, 3):
          map_crops['POD_%s%s' % (l,i)] = [pachymetry_od_crops['%s%s' % (l,i)][0], pachymetry_od_crops['%s%s' % (l,i)][1], L, A,'0123456789']

      for l in lista:
        for i in range(1, 3):
          map_crops['POS_%s%s' % (l,i)] = [pachymetry_os_crops['%s%s' % (l,i)][0], pachymetry_os_crops['%s%s' % (l,i)][1], L, A,'0123456789']

      for l in lista:
        for i in range(1, 3):
          map_crops['EOD_%s%s' % (l,i)] = [epithelium_od_crops['%s%s' % (l,i)][0], epithelium_od_crops['%s%s' % (l,i)][1], L, A,'0123456789']

      for l in lista:
        for i in range(1, 3):
          map_crops['EOS_%s%s' % (l,i)] = [epithelium_os_crops['%s%s' % (l,i)][0], epithelium_os_crops['%s%s' % (l,i)][1], L, A,'0123456789']
    else:
      map_crops = {
        'CO_POS': [pachymetry_os_cropsp2['CO'][0], pachymetry_os_cropsp2['CO'][1], L, A,'0123456789'],
        'CO_EOS': [epithelium_os_cropsp2['CO'][0], epithelium_os_cropsp2['CO'][1], L, A,'0123456789']
      }

      for l in lista:
        for i in range(1, 3):
          map_crops['POS_%s%s' % (l,i)] = [pachymetry_os_cropsp2['%s%s' % (l,i)][0], pachymetry_os_cropsp2['%s%s' % (l,i)][1], L, A,'0123456789']

      for l in lista:
        for i in range(1, 3):
          map_crops['EOS_%s%s' % (l,i)] = [epithelium_os_cropsp2['%s%s' % (l,i)][0], epithelium_os_cropsp2['%s%s' % (l,i)][1], L, A,'0123456789']
          
    return map_crops


  except Exception as e:
    print('erro get_map_crops', e)


def classifica_valor(texto):
    texto_limpo = texto.strip()
    if re.fullmatch(r'\d+', texto_limpo):
        return "somente numero"
    else:
        return "misto"


def getting_infos_data(row_data, img, pontos):
    try:
        paciente_base = img.filename.split('\\')[-2] + '___' + img.filename.split('\\')[-1].split('__')[0]
        for k, v in pontos.items():
            img_cropped = img.crop((v[0], v[1], v[0]+v[2], v[1]+v[3]))
            img_cropped = resize_img(img_cropped.convert('L'), IMG_MAG)
            img_cropped = img_cropped.filter(ImageFilter.BoxBlur(IMG_BLUR))
            ocr_result = reader.readtext(np.array(img_cropped), paragraph=True, min_size=2, allowlist=v[4])

            if len(ocr_result) > 0:
                valor = ocr_result[0][1]
                row_data[k] = valor
                valor_limpo = valor.replace('/', '-').replace('(', '_').replace(')', '_').replace(',', '_') or '-'
                img_cropped.save(os.path.join(diretorio_raiz, RECORTES_DIR, paciente_base + '_' + k + '_valor_' + valor_limpo + '.jpg'))
            else:
                row_data[k] = ''
        return row_data
    except Exception as e:
        print('erro getting_infos_data', e)
        return row_data


def getting_exam_data(row_data, img, pontos):
    try:
        paciente_base = img.filename.split('\\')[-2] + '___' + img.filename.split('\\')[-1].split('__')[0]
        for k, v in pontos.items():
            img_cropped = img.crop((v[0], v[1], v[0]+v[2], v[1]+v[3]))
            img_cropped = resize_img(img_cropped.convert('L'), IMG_MAG, BIN_THRESHOLD)
            img_cropped = img_cropped.filter(ImageFilter.BoxBlur(IMG_BLUR))
            ocr_result = reader.readtext(np.array(img_cropped), min_size=2, allowlist=v[4])

            if len(ocr_result) > 0:
                valor = ocr_result[0][1]
                conf = ocr_result[0][2]
                row_data[k] = valor
                row_data[k + '_conf'] = conf

                valor_limpo = valor.replace('/', '-').replace('(', '_').replace(')', '_').replace(',', '_') or '-'
                img_cropped.save(os.path.join(diretorio_raiz, RECORTES_DIR, paciente_base + '_' + k + '_valor_' + valor_limpo + '.jpg'))
            else:
                row_data[k] = ''
                row_data[k + '_conf'] = 0
        return row_data
    except Exception as e:
        print('erro getting_exam_data', e)
        return row_data
    

def getting_maps_data(row_data, img, pontos):
    try:
        paciente_base = img.filename.split('\\')[-2] + '___' + img.filename.split('\\')[-1].split('__')[0]
        for k, v in pontos.items():
            img_cropped = img.crop((v[0], v[1], v[0]+v[2], v[1]+v[3]))
            img_cropped = resize_img(img_cropped, IMG_MAG)
            ocr_result = reader.readtext(np.array(img_cropped), min_size=5, allowlist=v[4])

            if len(ocr_result) > 0:
                valor = ocr_result[0][1]
                conf = ocr_result[0][2]
                row_data[k] = valor
                row_data[k + '_conf'] = conf

                valor_limpo = valor.replace('/', '-').replace('(', '_').replace(')', '_').replace(',', '_') or '-'
                img_cropped.save(os.path.join(diretorio_raiz, RECORTES_DIR, paciente_base + '_' + k + '_valor_' + valor_limpo + '.jpg'))          
            else:
                row_data[k] = ''
                row_data[k + '_conf'] = 0
        return row_data
    except Exception as e:
        print('erro getting_maps_data', e)
        return row_data


def separa_dados_olhos(dados):
  try:
    dados_os = {}
    dados_od = {}
    for d in dados.keys():
      if 'OS' not in d and 'OD' not in d:
        if d == 'Eye':
          dados_od[d] = dados[d]
        elif d == 'Eye_2':
          dados_os[d] = dados[d]
        else:
          dados_os[d] = dados[d]
          dados_od[d] = dados[d]
      elif 'OS' in d:
        dados_os[d] = dados[d]
      elif 'OD' in d:
        dados_od[d] = dados[d]
    
    return [dados_os, dados_od]
  except Exception as e:
    print(e)    


def create_dataframe(dados):
  try:
    print('criando dataframe...')

    #lista os diretorios carregados
    lista_diretorios = []
    d = ''
    for dado in dados:
      if d != dado[0]:
        lista_diretorios.append(dado[0])
        d = dado[0]
      
    for d in lista_diretorios:
      dictionary_list = []
      dictionary_list_od = []
      dictionary_list_os = []
      df_final = None
      for dado in dados:
        if dado[0] == d:
          #verifica se os dados são dos dois olhos ou somente de um
          if 'Eye_2' in dado[1]: #Dois olhos
            dados_olhos = separa_dados_olhos(dado[1])
            dictionary_list_os.append(dados_olhos[0])
            dictionary_list_od.append(dados_olhos[1])
          else: #apenas um olho
            dictionary_list.append(dado[1])

      #Se a lista 'dictionary_list_os' não for vazia significa que os dados em 
      #questão são dos dois olhos, entao sao criadas as planilhas separadas, uma para cada olho
      if len(dictionary_list_os) > 0:
        df_final = pd.DataFrame.from_dict(dictionary_list_os)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df_final = df_final.apply(pd.to_numeric, errors='ignore')
        df_final.to_excel(os.path.join(diretorio_raiz, OUTPUT_DIR, d+'_OS_RTVue_'+timestr+'.xlsx'))

        df_final = None
        df_final = pd.DataFrame.from_dict(dictionary_list_od)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df_final = df_final.apply(pd.to_numeric, errors='ignore')
        df_final.to_excel(os.path.join(diretorio_raiz, OUTPUT_DIR, d+'_OD_RTVue_'+timestr+'.xlsx'))

      #Nesse caso a planilha é criada normalmente
      else:
        df_final = pd.DataFrame.from_dict(dictionary_list)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df_final = df_final.apply(pd.to_numeric, errors='ignore')
        df_final.to_excel(os.path.join(diretorio_raiz, OUTPUT_DIR, d+'_RTVue_'+timestr+'.xlsx'))

  except Exception as e:
    print(e)
    return False

  return True


def listar_arquivos(diretorio_exames):
  lista_arquivos = []
  try:
    diretorio_imagens = os.path.join(diretorio_raiz, diretorio_exames)
    for root, dirs, files in os.walk(diretorio_imagens):
      for d in dirs:
        os.chdir(os.path.join(diretorio_imagens, d))
        for f in os.listdir():
          lista_arquivos.append([os.path.join(diretorio_imagens, d, f), d])

    return lista_arquivos
  except Exception as e:
    print('Erro ao listar os arquivos: ', e)


def padrao_imagem(img):
  try:
    img_cropped = img.crop((pachymetry_od_crops['CO'][0], pachymetry_od_crops['CO'][1], pachymetry_od_crops['CO'][0] + L, pachymetry_od_crops['CO'][1] + A))
    img_cropped = resize_img(img_cropped, IMG_MAG)
    ocr_result = reader.readtext(np.array(img_cropped), min_size=5, allowlist='0123456789')
    
    if ocr_result[0][1] != '':
      return 1
    else:
      return 2
  except Exception as e:
    return 2


def extrair_infomacoes_arquivo(row_data, arquivo):

  padrao = 0
  try:
    print('Processando o arquivo "%s..."' % arquivo[0].split('\\')[-1][:50])
    img = Image.open(os.path.join(diretorio_raiz, IMAGES_DIR, arquivo[0]))
    fn_splits = arquivo[0].split("_")
    row_data['fName'] = fn_splits[1]+', '+fn_splits[2]+' '+fn_splits[3]
    row_data['fID'] = fn_splits[4]
    row_data['fEye'] = fn_splits[6]
    row_data['fExameDate'] = fn_splits[7]
    row_data['fExameTime'] = fn_splits[8].replace("-", ":")
    row_data['fSex'] = fn_splits[9]
    row_data['fDOB'] = fn_splits[10]

    padrao = padrao_imagem(img)

    row_data = getting_infos_data({}, img, info_crops(padrao))
    row_data = getting_exam_data(row_data, img, exam_crops(padrao))
    row_data = getting_maps_data(row_data, img, get_map_crops(padrao))

    print('processamento do arquivo "%s..." finalizado' % arquivo[0].split('\\')[-1][:50])
    
  except Exception as e:
    print('falha ao tentar ler o arquivo "%s..."' % arquivo[0].split('\\')[-1][:50], e)
    
  return [arquivo[1], row_data]


def iniciar_processo_leitura_imagens(diretorio_exames):
    try:
        arquivos = listar_arquivos(diretorio_exames)
        resultado = [extrair_infomacoes_arquivo({}, arquivo) for arquivo in arquivos]

        create_dataframe(resultado)
    except Exception as e:
        print(e)


def SarmentoOCR(diretorio_exames):
    global reader, diretorio_raiz
    try:
        reader = ocr.Reader(['pt'], gpu=True)
        diretorio_raiz = os.path.dirname(__file__)
        iniciar_processo_leitura_imagens(diretorio_exames)
    except Exception as e:
        print(e)


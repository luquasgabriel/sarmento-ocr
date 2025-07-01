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


OCR_CUTOFF = 0.6          # Limite mínimo de confiança para considerar um resultado do OCR
IMG_MAG = 4               # Fator de ampliação da imagem antes do OCR (aumenta a legibilidade)
BIN_THRESHOLD = 200       # Valor de limiar para binarização da imagem (0-255)
IMG_BLUR = 1              # Intensidade do desfoque aplicado antes da binarização
VERBOSE = False           # Ativa/desativa saída detalhada no console
save_step = 500           # Número de imagens processadas antes de salvar os resultados parciais
OUTPUT_DIR = "output"     # Diretório onde os arquivos de saída serão salvos
IMAGES_DIR = "imagens"    # Diretório de entrada contendo as imagens a serem processadas
RECORTES_DIR = 'recortes' # Diretório onde serão salvas imagens recortadas (regiões de interesse)

dictionary_list_p1 = []   # Lista de dicionários da primeira etapa de extração
dictionary_list_p2 = []   # Lista de dicionários da segunda etapa de extração (se houver)
row_data = {}             # Dicionário temporário para armazenar dados de uma imagem específica
ground_truth = {}         # Dicionário para armazenar os valores esperados de cada variável


'''
Leitura da "ground truth" (valores esperados) a partir de um arquivo CSV.
Esses dados servirão como base de comparação ou validação para os resultados do OCR.
'''
with open('ground_truth.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        chave = row['variavel'].strip()
        valor = row['valor'].strip()
        ground_truth[chave] = valor


'''
Função de binarização de imagem.
Converte a imagem para tons de cinza e aplica um limiar para transformá-la em uma imagem binária (preto e branco).
Essa etapa é importante para melhorar a performance do OCR ao eliminar ruído e focar no contraste entre texto e fundo.
'''
def binarize(img, thresh=200):  
  img=img.convert('L') # Converte a imagem para escala de cinza
  width,height=img.size

  for x in range(width):  #Varre os pixels da imagem
    for y in range(height):

      if img.getpixel((x,y)) < thresh:
        img.putpixel((x,y),0) # Se intensidade menor que o limiar, atribui preto
      else:
        img.putpixel((x,y),255) #Se intensidade maior ou igual ao limiar, atribui branco
  return img


'''
Função para recortar automaticamente as bordas vazias de uma imagem.
Utiliza a diferença entre a imagem e seu plano de fundo para identificar a região com conteúdo relevante.
'''
def trim_img(im):
  bg = Image.new(im.mode, im.size, im.getpixel((0,0))) # Cria uma imagem de fundo com a mesma cor do canto superior esquerdo
  diff = ImageChops.difference(im, bg)  # Calcula a diferença entre a imagem original e o fundo
  diff = ImageChops.add(diff, diff, 2.0, -100) # Aumenta o contraste da diferença para facilitar a detecção das bordas
  bbox = diff.getbbox() # Obtém o menor retângulo (bounding box) que contém todos os pixels diferentes do fundo
  if bbox:
    return im.crop(bbox)


'''
Redimensiona a imagem com base em um fator de ampliação.
'''
def resize_img(img, magnification, binarize_thresh=-1):
  width, height = img.size
  im = img.resize((width*magnification,height*magnification), Image.Resampling.LANCZOS)
  if binarize_thresh != -1:
    im = binarize(im, thresh=binarize_thresh)
  return im


'''
Função que retorna os recortes (crops) de regiões específicas da imagem com base em um padrão de layout.
Cada entrada no dicionário define uma região de interesse com as seguintes informações: [x, y, largura, altura, caracteres válidos esperados]
Usado para extrair campos como nome do paciente, data do exame, olho examinado, etc.
'''
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


'''
Função que define os recortes das regiões contendo dados dos exames oftalmológicos.
A estrutura varia conforme o layout identificado pelo parâmetro `padrao`.
Cada campo é representado por: [x, y, largura, altura, caracteres esperados]
Esses recortes são usados para extrair informações como curvaturas, espessuras, potências e métricas epiteliais via OCR.
'''
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


'''
Definição das regiões (coordenadas) onde o OCR deve extrair valores específicos de mapas de espessura da córnea
e da epitélio, tanto para o olho direito (OD) quanto para o olho esquerdo (OS), em dois padrões de layout diferentes.
'''

# Variáveis de dimensão padrão dos recortes utilizados
L = 45 # Largura
A = 15 # Altura

pachymetry_od_crops = { 
  'CO': [270,230],  # Centro óptico
  'S1': [268,146],  # Região superior
  'S2': [268,98],
  'SN1': [328,172], # Superior-nasal
  'SN2': [362,134],
  'N1': [354,230],  # Nasal
  'N2': [403,231],
  'IN1': [328,289], # Inferior-nasal
  'IN2': [364,325],
  'I1': [270,314],  # Inferior
  'I2': [269,362],
  'IT1': [211,290], # Inferior-temporal
  'IT2': [173,324],
  'T1': [184,230],  # Temporal
  'T2': [141,230],
  'ST1': [210,171], # Superior-temporal   
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

# Dicionários com coordenadas para o padrão 2
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

# Lista representando as regiões periféricas usadas em análises específicas ou ordenação dos pontos
lista = ['S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']


'''
Função que constrói um dicionário com os recortes (coordenadas e dimensões) das regiões dos mapas de paquimetria
e epitélio para serem processadas via OCR. O layout varia conforme o padrão do exame (1 ou 2).
As chaves indicam o tipo de mapa (POD, POS, EOD, EOS), a região (ex: 'S1', 'IN2'), e o olho (OD = olho direito, OS = esquerdo).
A estrutura de cada item: [x, y, largura, altura, caracteres esperados].
'''
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


'''
Funções que processam uma imagem para extrair informações via OCR a partir de pontos recortados pré-definidos.
Para cada região (definida em 'pontos'), a função:
- recorta a imagem;
- redimensiona e aplica filtro de desfoque para melhorar OCR;
- realiza OCR considerando uma lista de caracteres permitidos;
- salva a imagem do recorte com o valor extraído no nome;
- armazena o valor extraído no dicionário row_data.
'''
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


'''
Função para separar os dados extraídos em dois dicionários distintos: um para o olho esquerdo (OS) e outro para o olho direito (OD).
Caso a chave não contenha 'OS' nem 'OD', distribui os dados conforme regras específicas:
- 'Eye' vai para olho direito (OD)
- 'Eye_2' vai para olho esquerdo (OS)
- Outras chaves sem 'OS' ou 'OD' são atribuídas a ambos os olhos.
'''
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


'''
Função que cria DataFrames pandas a partir dos dados extraídos e os salva em arquivos Excel.
Para cada diretório base em 'dados', ela agrupa os dados correspondentes,
verifica se os dados contêm informações para ambos os olhos ou apenas um,
e então gera uma ou duas planilhas Excel (separadas por olho) no diretório de saída.
'''
def create_dataframe(dados):
  try:
    print('Criando dataframe...')

    # Lista dos diretórios base extraídos da primeira posição dos elementos em 'dados'
    lista_diretorios = []
    d = ''
    for dado in dados:
      if d != dado[0]:
        lista_diretorios.append(dado[0])
        d = dado[0]
      
    # Processa dados por diretório base
    for d in lista_diretorios:
      dictionary_list = []      # Para dados de um olho só
      dictionary_list_od = []   # Dados para olho direito (OD)
      dictionary_list_os = []   # Dados para olho esquerdo (OS)
      df_final = None

      # Separa os dados daquele diretório em listas específicas
      for dado in dados:
        if dado[0] == d:
          # Se dados para ambos os olhos estão presentes, separa-os
          if 'Eye_2' in dado[1]:
            dados_olhos = separa_dados_olhos(dado[1])
            dictionary_list_os.append(dados_olhos[0])
            dictionary_list_od.append(dados_olhos[1])
          else:
            dictionary_list.append(dado[1])

      # Caso haja dados para ambos os olhos, cria planilhas separadas para OS e OD
      if len(dictionary_list_os) > 0:
        # DataFrame olho esquerdo
        df_final = pd.DataFrame.from_dict(dictionary_list_os)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df_final = df_final.apply(pd.to_numeric, errors='ignore')  # tenta converter valores para numérico
        df_final.to_excel(os.path.join(diretorio_raiz, OUTPUT_DIR, d + '_OS_RTVue_' + timestr + '.xlsx'))

        # DataFrame olho direito
        df_final = None
        df_final = pd.DataFrame.from_dict(dictionary_list_od)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df_final = df_final.apply(pd.to_numeric, errors='ignore')
        df_final.to_excel(os.path.join(diretorio_raiz, OUTPUT_DIR, d + '_OD_RTVue_' + timestr + '.xlsx'))

      else:
        # Caso dados sejam de um olho só, gera apenas um arquivo Excel
        df_final = pd.DataFrame.from_dict(dictionary_list)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df_final = df_final.apply(pd.to_numeric, errors='ignore')
        df_final.to_excel(os.path.join(diretorio_raiz, OUTPUT_DIR, d + '_RTVue_' + timestr + '.xlsx'))

  except Exception as e:
    print(e)
    return False

  return True


'''
Função para listar arquivos de exames em um diretório base, percorrendo subpastas.
Para cada arquivo encontrado nas subpastas, adiciona uma lista com:
- o caminho completo do arquivo
- o nome da subpasta onde o arquivo está localizado
'''
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


'''
Função para identificar o padrão da imagem (padrão 1 ou 2) com base no resultado do OCR em uma região específica.
Recorta uma área fixa na imagem relacionada à paquimetria do olho direito ('CO' em pachymetry_od_crops),
redimensiona essa área, realiza OCR buscando números,
e decide o padrão:
- Se o OCR extrai algum texto não vazio, retorna 1 (padrão 1).
- Caso contrário, retorna 2 (padrão 2).
Em caso de erro (ex: OCR falha), também retorna 2 como padrão padrão.
'''
def padrao_imagem(img):
  try:
    img_cropped = img.crop((pachymetry_od_crops['CO'][0], pachymetry_od_crops['CO'][1], pachymetry_od_crops['CO'][0] + L, pachymetry_od_crops['CO'][1] + A))
    img_cropped = resize_img(img_cropped, IMG_MAG)
    ocr_result = reader.readtext(np.array(img_cropped), min_size=5, allowlist='0123456789')  # Executa OCR, buscando apenas dígitos numéricos
    
    if ocr_result[0][1] != '':
      return 1
    else:
      return 2
  except Exception as e:
    return 2


'''
Função que extrai informações de um arquivo de exame oftalmológico.
Recebe um dicionário inicial (row_data) e uma lista "arquivo" onde o primeiro elemento é o caminho do arquivo
e o segundo é o nome da subpasta/diretório.
A função:
- abre a imagem;
- extrai dados do nome do arquivo para preencher metadados (nome, ID, olho, data, sexo, data nascimento);
- detecta o padrão da imagem (padrão 1 ou 2);
- extrai dados específicos da imagem por OCR, utilizando funções auxiliares para diferentes partes;
- retorna uma lista com o nome da subpasta e o dicionário com os dados extraídos.
'''
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


'''
Função principal para iniciar o processo de leitura das imagens de exames.
Recebe o diretório base onde as imagens estão armazenadas.
Realiza os seguintes passos:
- Lista todos os arquivos e suas subpastas dentro do diretório informado.
- Para cada arquivo listado, extrai as informações usando OCR e outras funções auxiliares.
- Armazena os resultados em uma lista.
- Cria dataframes a partir dos dados extraídos e salva-os em arquivos Excel.
'''
def iniciar_processo_leitura_imagens(diretorio_exames):
    try:
        arquivos = listar_arquivos(diretorio_exames)
        resultado = [extrair_infomacoes_arquivo({}, arquivo) for arquivo in arquivos]

        create_dataframe(resultado)
    except Exception as e:
        print(e)


'''
Função principal que configura o OCR e inicia o processo de leitura das imagens.
Parâmetros:
- diretorio_exames: caminho para o diretório contendo as imagens dos exames.
Passos:
- Inicializa o leitor OCR da biblioteca EasyOCR para a língua portuguesa, utilizando GPU se disponível.
- Define o diretório raiz como o diretório onde este script está localizado.
- Chama a função que lista e processa as imagens presentes no diretório especificado.
'''
def SarmentoOCR(diretorio_exames):
    global reader, diretorio_raiz
    try:
        reader = ocr.Reader(['pt'], gpu=True)
        diretorio_raiz = os.path.dirname(__file__)
        iniciar_processo_leitura_imagens(diretorio_exames)
    except Exception as e:
        print(e)


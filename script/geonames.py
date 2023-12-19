# Импорт библиотек
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine.url import URL

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

from fuzzywuzzy import fuzz
from fuzzywuzzy import process # расстояние Левенштейна

from sentence_transformers import SentenceTransformer, util
import torch

from sklearn.metrics import accuracy_score


# Ввод функций и констант 

# Пароль на серевер
password_sql = ''

# Функция создания словаря из списка и ключа. Для преобразования вложенного списка в плоский
def alternatenames_dict(text, geonameid):
    list_text = text.split(',')
    dict_text = {}
    for i in list_text:
        dict_text[i] = geonameid
    return dict_text
    
# Функция для очистки текста
def clear_text(text):
    text = text.lower()
    t_str = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9-]', " ", text)
    t_str = " ".join(t_str.split())
    return t_str

# Функция для транслитирования текста из кириллицы в латинницу
def translit(text):
    legend = {
        'а': 'a',
        'б': 'b',
        'в': 'v',
        'г': 'g',
        'д': 'd',
        'е': 'e',
        'ё': 'yo',
        'ж': 'zh',
        'з': 'z',
        'и': 'i',
        'й': 'y',
        'к': 'k',
        'л': 'l',
        'м': 'm',
        'н': 'n',
        'о': 'o',
        'п': 'p',
        'р': 'r',
        'с': 's',
        'т': 't',
        'у': 'u',
        'ф': 'f',
        'х': 'h',
        'ц': 'ts',
        'ч': 'ch',
        'ш': 'sh',
        'щ': 'shch',
        'ъ': 'y',
        'ы': 'y',
        'ь': "'",
        'э': 'e',
        'ю': 'yu',
        'я': 'ya',
        }
    new_text = ''
    text = text.lower()
    for i in text:
        if i in legend:
            new_text += legend[i]
        elif i == ' ':
            new_text += ' '
        else:
            new_text += i

    return new_text


# Подготовка к загрузке данных в SQL
def sql(password_sql):
    DATABASE = {
        'drivername': 'postgresql',
        'username': 'postgres', 
        'password': password_sql, 
        'host': 'localhost',
        'port': '5432',
        'database': 'postgres',
        'query': {}
    }  
    engine = create_engine(URL(**DATABASE))
    return engine


# Функция расчета косинусного расстояния
def cos_dist(questi, word):
    query_vec = vectorizer_c.transform([clear_text(translit(questi))]).toarray()[0]
    word_vect = vectorizer_c.transform([clear_text(word)]).toarray()[0]
    distances = 1-distance.cosine(word_vect, query_vec)
    return distances


# Коды исследуемых стран
country_code = ['RU', 'BY', 'AM', 'KZ', 'KG', 'RS', 'TR']

# Названия исследуемых стран
country_name = {'RU': 'Russia', 
                'BY': 'Belarus', 
                'AM': 'Armenia', 
                'KZ': 'Kazakhstan', 
                'KG': 'Kyrgyzstan', 
                'RS': 'Serbia', 
                'TR': 'Turkey'}




class preparation_geoname(object)

    def __init__(self):
      
      
    # Выгрузка данных
    # Датасет с различными гео-показателями о всех городах с населением 15 000+ или столицах

    def cities1500(self):
        # Выгрузка таблицы с сервера
        engine = sql()
        query = """
        SELECT * 
        FROM cities1500 
        """
        df_cities1500 = pd.read_sql_query(text(query), con=engine.connect())
        # Оставим только исследуемые страны
        df_cities1500 = df_cities1500.query('`country code` in @country_code').reset_index(drop=True)
        # Определим список исследуемых geonameid
        geonameid_list = df_cities1500['geonameid'].unique()
        # Сократим число столбцов
        df_cities1500 = df_cities1500.loc[:, ['geonameid', 'name', 'alternatenames', 'country code', 'admin1 code', 'population']].copy()
        # Удалим пропуски 
        df_cities1500 = df_cities1500.dropna(subset=['alternatenames'])
        # Добавим столбец со словарем аоьтернативных названий
        df_cities1500['dict_text'] = df_cities1500.apply(lambda x: alternatenames_dict(x['alternatenames'], x['geonameid']), axis=1)
        # Объединим словари
        big_list = list(df_cities1500['dict_text'])
        big_diht = {}
        for i in big_list:
            big_diht.update(i)
        # Создадим новый дата-фрейм
        df_altern = pd.DataFrame(list(big_diht.items()))
        df_altern.columns = ['alternatenames', 'geonameid']
        return df_altern


    # Обучающий датасет с альтернативными именами и кодами языков
    def alternatenames_func(self):
        # Выгрузка таблицы с сервера
        df_altern = self.cities1500()
        engine = sql()
        query = """
        SELECT * 
        FROM public."alternateNamesV2"
        """
        df_alternateNamesV2 = pd.read_sql_query(text(query), con=engine.connect())
        # Создадим вторую часть датафейма альтернативных названий
        df_altern_alternateNamesV2 = df_alternateNamesV2.loc[:, ['geonameid', 'alternate name']].copy()
        df_altern_alternateNamesV2 = df_altern_alternateNamesV2.drop_duplicates().reset_index(drop=True)
        df_altern_alternateNamesV2.columns = ['geonameid', 'alternatenames']
        # Объединим датафреймы
        df_altern_all = pd.concat([df_altern, df_altern_alternateNamesV2], keys=['alternatenames', 'geonameid']).reset_index(drop=True)
        # Удалим строки, содержащие ссылки
        df_altern_all = df_altern_all[~df_altern_all.alternatenames.str.contains('https://')].copy()
        # Наименования приведем к нижниму регистру и очистим от лишних символов
        df_altern_all['alternatename'] = df_altern_all['alternatenames'].apply(clear_text)
        df_altern_all.drop('alternatenames', axis=1, inplace=True)
        df_altern_all = df_altern_all.drop_duplicates().reset_index(drop=True)
        # Создадим столбец с транслитированными русскими названиями
        df_altern_all['alternatename_1'] = df_altern_all['alternatename'].apply(translit)
        df_altern_all['alternatename_en'] = df_altern_all['alternatename_1'].apply(clear_text)
        df_altern_all.drop('alternatename_1', axis=1, inplace=True)
        return df_altern_all


    def fit_alternem(self):
        # Создание обучающего датафрейма на сервере
        df_altern_all = self.alternatenames_func()
        df_altern_all.to_sql('altern_all', con=engine)





# Оценка

class predict_geoname(object)

    def __init__(self, num=1):
        self.num = num
        

    # Выгрузка таблиц с сервера

    def exstraction_cities1500(self):

        # Датасет с различными гео-показателями о всех городах с населением 15 000+ или столицах
        engine = sql()
        query = """
        SELECT * 
        FROM cities1500 
        """
        df_cities1500 = pd.read_sql_query(text(query), con=engine.connect())
        return df_cities1500

    def exstraction_admin1CodesASCII(self):
        # Названия административных подразделений на английском языке
        engine = sql()
        query = """
        SELECT * 
        FROM public."admin1CodesASCII"
        """
        df_admin1CodesASCII = pd.read_sql_query(text(query), con=engine.connect())
        return df_admin1CodesASCII

    def exstraction_altern_all(self):   
        # Обучающющий датафрейм
        engine = sql()
        query = """
        SELECT * 
        FROM altern_all
        """
        df_altern_all = pd.read_sql_query(text(query), con=engine.connect())
        return df_altern_all


    def learn_corp(self):
        
        df_altern_all = self.exstraction_altern_all()
        # Обучающий корпус
        corpus = pd.Series(df_altern_all['alternatename'])
        corpus_en = pd.Series(df_altern_all['alternatename_en'])
        # Целевой показатель
        answ = pd.Series(df_altern_all['geonameid'])
        return [corpus, corpus_en, answ]

    # Евклидово расстояние для латиницы
    # База для формирования ответов

    def vector_en(self):
        corpus_en = self.learn_corp()[1]
        # создаем мешок слов
        vectorizer_en = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
        bow_en = vectorizer_en.fit_transform(corpus_en)
        return [vectorizer_en, bow_en]

    # Функция для формирования ответов
    def answ_predict(self, text):
        
        df_cities1500 = self.exstraction_cities1500()
        df_admin1CodesASCII = self.exstraction_admin1CodesASCII()
        answ = self.learn_corp()[2]
        
        vectorizer_en = self.vector_en()[0]
        bow_en = self.vector_en()[1]

        # Преобразуем входную строку в вектор 
        questi = pd.Series(clear_text(translit(text)))
        query_vec = vectorizer_en.transform(questi)

        # вычисляем евклидово расстояние между новой строкой и всеми строками в мешке слов
        distances = euclidean_distances(query_vec, bow_en)

        # получаем индекс наиболее близкой строки
        top_indices = distances.argsort()[0][:self.num]

        # Создаем списки данных
        geonameid_list = []
        name_list = []
        region_list =[]
        country_list = []
        cosine_list =[]

        for tops in range(self.num):
            # Ответ geonameid
            answ_geonameid = answ[int(top_indices[tops])]
            geonameid_list.append(answ_geonameid)

            # Ответ name
            answ_name = df_cities1500.loc[df_cities1500['geonameid']==answ_geonameid, 'name'].to_string(index=False)
            name_list.append(answ_name)

            # Ответ регион
            cc = df_cities1500.loc[df_cities1500['geonameid']==answ_geonameid, 'country code'].to_string(index=False)
            rc = df_cities1500.loc[df_cities1500['geonameid']==answ_geonameid, 'admin1 code'].to_string(index=False)
            ar = cc +'.'+ rc
            answ_region = df_admin1CodesASCII.loc[df_admin1CodesASCII['code']==ar, 'name ascii'].to_string(index=False)
            region_list.append(answ_region)

            # Ответ страна
            answ_country = country_name[df_cities1500.loc[df_cities1500['geonameid']==answ_geonameid, 'country code'].
                                        to_string(index=False)]
            country_list.append(answ_country)

            # Ответ cosine similarity
            cosine_similarity = cos_dist(text, corpus_en[top_indices[tops]])
            cosine_list.append(cosine_similarity)

        df_answ = pd.DataFrame({
            'geonameid': geonameid_list,
            'name': name_list,
            'region': region_list,
            'country': country_list,
            'cosine': cosine_list
        })
        return df_answ.to_dict(orient='records')


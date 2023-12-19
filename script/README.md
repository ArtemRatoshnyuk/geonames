# Описание работы сервиса geonames

## Запустить:

Для обработки `from geonames import preparation_geoname`
Для получения ответа `from geonames import predict_geoname`

**До начала со скриптом ввести пароль для сервера** (переменная `password_sql`)

## Скрипт содержит два метода:

### Обработка файлов для последующего дообучения модели

Для запуска обработки необходимо вызвать в классе `preparation_geoname` функцию `fit_alternem()`

### Выведение списка словарей, содержащих информацию о географическом названии

Для выведения ответа необходимо вызвать в классе `predict_geoname` функцию `answ_predict ()`

В качестве параметра класса можно ввести **количество строк**, которое будет выводиться в ответе (по умолчанию – одна строка). В качестве параметра функции необходимо ввести **название объекта** (тип – строка). 

**Пример ввода:**

`predict_geoname(5).answ_predict (‘Екатеринбург)`

**Пример ответа** (geonameid, name, region, country, cosine similarity):

`[{'geonameid': 1486209,
  'name': 'Yekaterinburg',
  'region': 'Sverdlovsk Oblast',
  'country': 'Russia',
  'cosine': 1.0},
 {'geonameid': 1486209,
  'name': 'Yekaterinburg',
  'region': 'Sverdlovsk Oblast',
  'country': 'Russia',
  'cosine': 1.0},
 {'geonameid': 1486209,
  'name': 'Yekaterinburg',
  'region': 'Sverdlovsk Oblast',
  'country': 'Russia',
  'cosine': 0.9252126882606606},
 {'geonameid': 1486209,
  'name': 'Yekaterinburg',
  'region': 'Sverdlovsk Oblast',
  'country': 'Russia',
  'cosine': 0.9252126882606606},
 {'geonameid': 1486209,
  'name': 'Yekaterinburg',
  'region': 'Sverdlovsk Oblast',
  'country': 'Russia',
  'cosine': 0.9252126882606606}]`



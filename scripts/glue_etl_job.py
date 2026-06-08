import sys, json, re, io
import pandas as pd
import pymysql
import boto3
from pymongo import MongoClient

# ── Parámetros ────────────────────────────────────────────────────────────────
S3_BUCKET        = 'mi-bucket-noticias-hito1'
S3_PREFIX_KAGGLE = 'raw/kaggle/'
S3_PREFIX_HF     = 'raw/huggingface/'
MONGO_URI        = 'mongodb+srv://victorramirezmunozal_db_user:mongo_pass@cluster0.rihar3a.mongodb.net/?appName=Cluster0'
MONGO_DB         = 'noticias_db'
MONGO_COL        = 'newsapi_raw'
RDS_HOST         = 'noticias-db-hito1.cbx7jeymu2xj.us-east-1.rds.amazonaws.com'
RDS_PORT         = 3306
RDS_DB           = 'noticias_hito1'
RDS_USER         = 'admin'
RDS_PASSWORD     = 'noticias_rds!'
RDS_TABLE        = 'raw_news_combined'

# ── Limpieza de texto ─────────────────────────────────────────────────────────
# PLACEHOLDER: añade aquí los pasos de limpieza que necesites.
# Actualmente sólo elimina tags HTML y colapsa espacios.
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)       # quitar HTML
    text = re.sub(r'\s+', ' ', text).strip()   # colapsar espacios
    return text

# ── Helpers S3 ────────────────────────────────────────────────────────────────
def list_s3_keys(bucket, prefix):
    s3 = boto3.client('s3')
    pag = s3.get_paginator('list_objects_v2')
    return [o['Key'] for p in pag.paginate(Bucket=bucket, Prefix=prefix) for o in p.get('Contents', [])]

def read_csv_s3(bucket, key):
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

def read_parquet_s3(bucket, key):
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

# ── 1. Leer Kaggle desde S3 ───────────────────────────────────────────────────
print('[Glue] Leyendo Kaggle desde S3...')
kaggle_frames = []
for key in list_s3_keys(S3_BUCKET, S3_PREFIX_KAGGLE):
    df = read_csv_s3(S3_BUCKET, key)
    df['label']  = 'Real' if 'True' in key else 'Fake'
    df['source'] = 'kaggle'
    kaggle_frames.append(df)
df_kaggle = pd.concat(kaggle_frames, ignore_index=True) if kaggle_frames else pd.DataFrame()
print(f'[Glue] Kaggle: {len(df_kaggle)} filas')

# ── 2. Leer HuggingFace desde S3 ─────────────────────────────────────────────
print('[Glue] Leyendo HuggingFace desde S3...')
hf_frames = []
for key in list_s3_keys(S3_BUCKET, S3_PREFIX_HF):
    df = read_parquet_s3(S3_BUCKET, key)
    hf_frames.append(df)
df_hf = pd.concat(hf_frames, ignore_index=True) if hf_frames else pd.DataFrame()
# df_hf ya trae columna 'label' con valores 0/1 — no sobreescribir
if not df_hf.empty:
    df_hf['source'] = 'huggingface'
print(f'[Glue] HuggingFace: {len(df_hf)} filas')

# ── 3. Leer NewsAPI desde MongoDB ─────────────────────────────────────────────
print('[Glue] Leyendo NewsAPI desde MongoDB...')
docs = list(MongoClient(MONGO_URI)[MONGO_DB][MONGO_COL].find({}, {'_id': 0}))
df_mongo = pd.DataFrame(docs)
df_mongo['label']  = None
df_mongo['source'] = 'newsapi'
print(f'[Glue] MongoDB: {len(df_mongo)} documentos')

# ── 4. Normalizar a esquema común ─────────────────────────────────────────────
def normalize(df, title_col, text_col, date_col, source):
    out = pd.DataFrame()
    out['title']         = df.get(title_col, pd.Series(dtype=str))
    out['content_clean'] = df.get(text_col, pd.Series(dtype=str)).apply(clean_text)
    out['published_at']  = df.get(date_col, pd.Series(dtype=str)).astype(str)
    out['label']         = df.get('label', pd.Series(dtype=str))
    if source == 'huggingface':
        out['label'] = out['label'].map({0: 'Fake', 1: 'Real'})
    out['source'] = source
    return out

frames = []
if not df_kaggle.empty: frames.append(normalize(df_kaggle, 'title', 'text',    'date',        'kaggle'))
if not df_hf.empty:     frames.append(normalize(df_hf,     'title', 'text',    'date',        'huggingface'))
if not df_mongo.empty:  frames.append(normalize(df_mongo,  'title', 'content', 'publishedAt', 'newsapi'))

df_final = pd.concat(frames, ignore_index=True)
df_final.dropna(subset=['title', 'content_clean'], inplace=True)
df_final.reset_index(drop=True, inplace=True)
print(f'[Glue] Total filas unificadas: {len(df_final)}')

# ── 5. Escribir en RDS ────────────────────────────────────────────────────────
print('[Glue] Escribiendo en RDS...')
conn = pymysql.connect(host=RDS_HOST, port=RDS_PORT, user=RDS_USER,
                       password=RDS_PASSWORD, database=RDS_DB, charset='utf8mb4',
                       ssl={'ssl': True})
with conn.cursor() as cur:
    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS `{RDS_TABLE}` (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            source        VARCHAR(50),
            title         TEXT,
            content_clean TEXT,
            published_at  VARCHAR(50),
            label         VARCHAR(20),
            ingested_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    ''')
    rows = [
        (str(r['source'])[:50], str(r['title'])[:500] if r['title'] else None,
         str(r['content_clean'])[:5000] if r['content_clean'] else None,
         str(r['published_at'])[:50],
         str(r['label'])[:20] if r['label'] and str(r['label']) != 'None' else None)
        for _, r in df_final.iterrows()
    ]
    cur.executemany(
        f'INSERT INTO `{RDS_TABLE}` (source,title,content_clean,published_at,label) VALUES (%s,%s,%s,%s,%s)',
        rows
    )
conn.commit()
conn.close()
print(f'[Glue] {len(rows)} filas escritas en RDS.')

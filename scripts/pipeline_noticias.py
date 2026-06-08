"""
=============================================================================
PIPELINE DE INGESTA DE NOTICIAS - HITO 1
=============================================================================
Flujo de datos:

  [Kaggle CSV]          ──────────────────────────┐
                                                  ├──► S3
  [HuggingFace Parquet] ──────────────────────────┘        \
                                                             ├──► AWS Glue (ETL) ──► RDS
  [NewsAPI] ──► Kafka (Confluent) ──► MongoDB Atlas ────────┘

=============================================================================
"""
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# SECCIÓN 1 — VARIABLES DE CONFIGURACIÓN  (edita aquí y sólo aquí)
# =============================================================================

# ── AWS Academy ───────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN     = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
 
# ── S3 ────────────────────────────────────────────────────────────────────────
S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME")
S3_PREFIX_KAGGLE      = os.getenv("S3_PREFIX_KAGGLE", "raw/kaggle/")
S3_PREFIX_HF          = os.getenv("S3_PREFIX_HF", "raw/huggingface/")
 
# ── Kaggle ────────────────────────────────────────────────────────────────────
KAGGLE_LOCAL_DIR      = os.getenv("KAGGLE_LOCAL_DIR", "./datasets/kaggle/")
KAGGLE_FILES          = os.getenv("KAGGLE_FILES", "True.csv,Fake.csv").split(",")
 
# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_DATASET_NAME       = os.getenv("HF_DATASET_NAME")
HF_DATASET_SPLIT      = os.getenv("HF_DATASET_SPLIT", "train")
HF_SAMPLE_SIZE        = int(os.getenv("HF_SAMPLE_SIZE", 50000))
HF_LOCAL_FILE         = os.getenv("HF_LOCAL_FILE", "./datasets/hf/fake_news_hf.parquet")
 
# ── NewsAPI ───────────────────────────────────────────────────────────────────
NEWSAPI_KEY           = os.getenv("NEWSAPI_KEY")
NEWSAPI_QUERY         = os.getenv("NEWSAPI_QUERY", "artificial intelligence")
NEWSAPI_LANGUAGE      = os.getenv("NEWSAPI_LANGUAGE", "en")
NEWSAPI_PAGE_SIZE     = int(os.getenv("NEWSAPI_PAGE_SIZE", 70))
 
# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP       = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_API_KEY         = os.getenv("KAFKA_API_KEY", "")
KAFKA_API_SECRET      = os.getenv("KAFKA_API_SECRET", "")
KAFKA_TOPIC           = os.getenv("KAFKA_TOPIC", "noticias-raw")
 
# ── MongoDB Atlas ─────────────────────────────────────────────────────────────
MONGO_URI             = os.getenv("MONGO_URI")
MONGO_DB_NAME         = os.getenv("MONGO_DB_NAME", "noticias_db")
MONGO_COLLECTION      = os.getenv("MONGO_COLLECTION", "newsapi_raw")
 
# ── Amazon RDS ────────────────────────────────────────────────────────────────
RDS_HOST              = os.getenv("RDS_HOST")
RDS_PORT              = int(os.getenv("RDS_PORT", 3306))
RDS_DB_NAME           = os.getenv("RDS_DB_NAME", "noticias_hito1")
RDS_USER              = os.getenv("RDS_USER", "admin")
RDS_PASSWORD          = os.getenv("RDS_PASSWORD")
RDS_TABLE             = os.getenv("RDS_TABLE", "raw_news_combined")
 
# ── AWS Glue ──────────────────────────────────────────────────────────────────
GLUE_JOB_NAME         = os.getenv("GLUE_JOB_NAME", "etl-noticias-hito1")
GLUE_SCRIPT_S3_PATH   = f"s3://{S3_BUCKET_NAME}/scripts/glue_etl_job.py"
GLUE_IAM_ROLE         = os.getenv("GLUE_IAM_ROLE")
GLUE_TEMP_DIR         = os.getenv("GLUE_TEMP_DIR", f"s3://{S3_BUCKET_NAME}/tmp/glue/")

# ── Control de flujo ──────────────────────────────────────────────────────────
# Pon en False las etapas que no quieras ejecutar en una corrida concreta
RUN_SETUP_S3          = False
RUN_INGEST_KAGGLE     = False
RUN_INGEST_HF         = False
RUN_INGEST_NEWSAPI    = False
RUN_KAFKA_PRODUCER    = False
RUN_INGEST_MONGO      = False
RUN_GLUE_UPLOAD       = False   # Ya creado manualmente
RUN_GLUE_CREATE_JOB   = False   # Ya subido manualmente
RUN_GLUE_START_JOB    = True

# =============================================================================
# SECCIÓN 2 — IMPORTS
# =============================================================================
import json
import logging
import os
import time
from datetime import datetime

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# SECCIÓN 3 — HELPERS COMPARTIDOS
# =============================================================================

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )


def get_glue_client():
    return boto3.client(
        "glue",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME][MONGO_COLLECTION]


# =============================================================================
# SECCIÓN 4 — SETUP S3
# =============================================================================

def setup_s3_bucket():
    """Crea el bucket de S3 si no existe."""
    log.info("── ETAPA 1: Configurando bucket S3 ──")
    s3 = get_s3_client()
    try:
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        log.info(f"  El bucket '{S3_BUCKET_NAME}' ya existe.")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=S3_BUCKET_NAME)
            else:
                s3.create_bucket(
                    Bucket=S3_BUCKET_NAME,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                )
            log.info(f"  Bucket '{S3_BUCKET_NAME}' creado.")
        else:
            raise


# =============================================================================
# SECCIÓN 5 — INGESTA BATCH: KAGGLE → S3
# =============================================================================

def ingest_kaggle_to_s3():
    """
    Sube los CSV de Kaggle al bucket S3.
    Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

    Pasos previos manuales:
      1. kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
      2. Descomprimir en KAGGLE_LOCAL_DIR
    """
    log.info("── ETAPA 2A: Kaggle → S3 ──")
    s3 = get_s3_client()
    os.makedirs(KAGGLE_LOCAL_DIR, exist_ok=True)

    for fname in KAGGLE_FILES:
        local_path = os.path.join(KAGGLE_LOCAL_DIR, fname)
        if not os.path.exists(local_path):
            log.warning(f"  Archivo no encontrado: {local_path}. Omitiendo.")
            continue
        s3_key = f"{S3_PREFIX_KAGGLE}{fname}"
        log.info(f"  Subiendo {fname} → s3://{S3_BUCKET_NAME}/{s3_key}")
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        log.info(f"  ✓ {fname} subido.")


# =============================================================================
# SECCIÓN 6 — INGESTA BATCH: HUGGING FACE → S3
# =============================================================================

def ingest_huggingface_to_s3():
    log.info("── ETAPA 2B: Hugging Face → S3 ──")
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("  Instala: pip install datasets pyarrow")
        return

    splits = ["train", "validation", "test"]
    all_rows = []

    for split in splits:
        log.info(f"  Descargando split '{split}'...")
        dataset = load_dataset(HF_DATASET_NAME, split=split, streaming=True)
        for row in dataset:
            all_rows.append(row)
        log.info(f"  Split '{split}' descargado. Total acumulado: {len(all_rows)} filas")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(HF_LOCAL_FILE), exist_ok=True)
    df.to_parquet(HF_LOCAL_FILE, index=False)
    log.info(f"  Guardado localmente: {HF_LOCAL_FILE} ({len(df)} filas)")

    s3 = get_s3_client()
    s3_key = f"{S3_PREFIX_HF}{os.path.basename(HF_LOCAL_FILE)}"
    s3.upload_file(HF_LOCAL_FILE, S3_BUCKET_NAME, s3_key)
    log.info(f"  ✓ Subido → s3://{S3_BUCKET_NAME}/{s3_key}")

# =============================================================================
# SECCIÓN 7 — INGESTA STREAMING: NEWSAPI → KAFKA → MONGODB
# =============================================================================

def fetch_newsapi_articles() -> list:
    """
    Descarga artículos de NewsAPI.
    Docs: https://newsapi.org/docs/endpoints/everything
    """
    log.info("── ETAPA 3A: Fetching NewsAPI ──")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        NEWSAPI_QUERY,
        "language": NEWSAPI_LANGUAGE,
        "pageSize": NEWSAPI_PAGE_SIZE,
        "sortBy":   "publishedAt",
        "apiKey":   NEWSAPI_KEY,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    log.info(f"  ✓ {len(articles)} artículos obtenidos.")
    return articles


def produce_to_kafka(articles: list):
    """
    Envía cada artículo al topic de Kafka como mensaje JSON.
    Confluent Cloud (tier gratuito): https://www.confluent.io/confluent-cloud/
    Si la librería no está instalada avisa y continúa.
    """
    log.info("── ETAPA 3B: Kafka Producer ──")
    try:
        from confluent_kafka import Producer
    except ImportError:
        log.warning("  confluent-kafka no instalado (pip install confluent-kafka). Saltando.")
        return

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
    })

    enviados = 0
    for article in articles:
        payload = json.dumps(article, ensure_ascii=False).encode("utf-8")
        producer.produce(
            KAFKA_TOPIC,
            value=payload,
            callback=lambda err, msg: log.error(f"  Kafka error: {err}") if err else None,
        )
        enviados += 1

    producer.flush()
    log.info(f"  ✓ {enviados} mensajes enviados al topic '{KAFKA_TOPIC}'.")


def save_articles_to_mongo(articles: list):
    """
    Inserta los artículos en MongoDB Atlas M0 (512 MB gratis).
    Registro: https://www.mongodb.com/cloud/atlas/register
    """
    log.info("── ETAPA 3C: MongoDB Atlas ──")
    ts = datetime.utcnow().isoformat()
    for art in articles:
        art["_ingested_at"] = ts

    collection = get_mongo_collection()
    if articles:
        result = collection.insert_many(articles)
        log.info(f"  ✓ {len(result.inserted_ids)} documentos insertados en MongoDB.")
    else:
        log.warning("  Sin artículos que insertar.")


# =============================================================================
# SECCIÓN 8 — AWS GLUE ETL
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Script que se ejecuta DENTRO de Glue.
# Lee S3 (Kaggle + HF) y MongoDB, limpia y escribe en RDS.
#
# La función clean_text() es el lugar donde debes añadir tu lógica de
# limpieza cuando la tengas definida. Por ahora hace lo mínimo seguro:
# quitar HTML y colapsar espacios.
# ─────────────────────────────────────────────────────────────────────────────

#Al final se usa glue_etl_job.py ya que tuvo que subirse de manera manual
GLUE_JOB_SCRIPT = (
    "import sys, json, re, io\n"
    "import pandas as pd\n"
    "import pymysql\n"
    "import boto3\n"
    "from pymongo import MongoClient\n"
    "\n"
    "# ── Parámetros (heredados del orquestador via variables de entorno) ──\n"
    f"S3_BUCKET        = '{S3_BUCKET_NAME}'\n"
    f"S3_PREFIX_KAGGLE = '{S3_PREFIX_KAGGLE}'\n"
    f"S3_PREFIX_HF     = '{S3_PREFIX_HF}'\n"
    f"MONGO_URI        = '{MONGO_URI}'\n"
    f"MONGO_DB         = '{MONGO_DB_NAME}'\n"
    f"MONGO_COL        = '{MONGO_COLLECTION}'\n"
    f"RDS_HOST         = '{RDS_HOST}'\n"
    f"RDS_PORT         = {RDS_PORT}\n"
    f"RDS_DB           = '{RDS_DB_NAME}'\n"
    f"RDS_USER         = '{RDS_USER}'\n"
    f"RDS_PASSWORD     = '{RDS_PASSWORD}'\n"
    f"RDS_TABLE        = '{RDS_TABLE}'\n"
    "\n"
    "# ── Limpieza de texto ────────────────────────────────────────────────\n"
    "# PLACEHOLDER: añade aquí los pasos de limpieza que necesites.\n"
    "# Actualmente sólo elimina tags HTML y colapsa espacios.\n"
    "def clean_text(text):\n"
    "    if not isinstance(text, str):\n"
    "        return ''\n"
    "    text = re.sub(r'<[^>]+>', ' ', text)   # quitar HTML\n"
    "    text = re.sub(r'\\s+', ' ', text).strip() # colapsar espacios\n"
    "    # --- AÑADE MÁS PASOS AQUÍ (stop-words, stemming, etc.) ---\n"
    "    return text\n"
    "\n"
    "# ── Helpers S3 ───────────────────────────────────────────────────────\n"
    "def list_s3_keys(bucket, prefix):\n"
    "    s3 = boto3.client('s3')\n"
    "    pag = s3.get_paginator('list_objects_v2')\n"
    "    return [o['Key'] for p in pag.paginate(Bucket=bucket, Prefix=prefix) for o in p.get('Contents', [])]\n"
    "\n"
    "def read_csv_s3(bucket, key):\n"
    "    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)\n"
    "    return pd.read_csv(obj['Body'])\n"
    "\n"
    "def read_parquet_s3(bucket, key):\n"
    "    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)\n"
    "    return pd.read_parquet(io.BytesIO(obj['Body'].read()))\n"
    "\n"
    "# ── 1. Leer Kaggle desde S3 ──────────────────────────────────────────\n"
    "print('[Glue] Leyendo Kaggle desde S3...')\n"
    "kaggle_frames = []\n"
    "for key in list_s3_keys(S3_BUCKET, S3_PREFIX_KAGGLE):\n"
    "    df = read_csv_s3(S3_BUCKET, key)\n"
    "    df['label']  = 'Real' if 'True' in key else 'Fake'\n"
    "    df['source'] = 'kaggle'\n"
    "    kaggle_frames.append(df)\n"
    "df_kaggle = pd.concat(kaggle_frames, ignore_index=True) if kaggle_frames else pd.DataFrame()\n"
    "print(f'[Glue] Kaggle: {len(df_kaggle)} filas')\n"
    "\n"
    "# ── 2. Leer HuggingFace desde S3 ─────────────────────────────────────\n"
    "print('[Glue] Leyendo HuggingFace desde S3...')\n"
    "hf_frames = []\n"
    "for key in list_s3_keys(S3_BUCKET, S3_PREFIX_HF):\n"
    "    df = read_parquet_s3(S3_BUCKET, key)\n"
    "    hf_frames.append(df)\n"
    "df_hf = pd.concat(hf_frames, ignore_index=True) if hf_frames else pd.DataFrame()\n"
    "# Nota: df_hf ya trae columna 'label' con valores 0/1 — no sobreescribir\n"
    "if not df_hf.empty: df_hf['source'] = 'huggingface'\n"
    "print(f'[Glue] HuggingFace: {len(df_hf)} filas')\n"
    "\n"
    "# ── 3. Leer NewsAPI desde MongoDB ────────────────────────────────────\n"
    "print('[Glue] Leyendo NewsAPI desde MongoDB...')\n"
    "docs = list(MongoClient(MONGO_URI)[MONGO_DB][MONGO_COL].find({}, {'_id': 0}))\n"
    "df_mongo = pd.DataFrame(docs)\n"
    "df_mongo['label']  = None\n"
    "df_mongo['source'] = 'newsapi'\n"
    "print(f'[Glue] MongoDB: {len(df_mongo)} documentos')\n"
    "\n"
    "# ── 4. Normalizar a esquema común ────────────────────────────────────\n"
    "def normalize(df, title_col, text_col, date_col, source):\n"
    "    out = pd.DataFrame()\n"
    "    out['title']         = df.get(title_col, pd.Series(dtype=str))\n"
    "    out['content_clean'] = df.get(text_col, pd.Series(dtype=str)).apply(clean_text)\n"
    "    out['published_at']  = df.get(date_col, pd.Series(dtype=str)).astype(str)\n"
    "    out['label']         = df.get('label', pd.Series(dtype=str))\n"
    "    if source == 'huggingface':\n"
    "        out['label'] = out['label'].map({0: 'Fake', 1: 'Real'})\n"
    "    out['source'] = source\n"
    "    return out\n"
    "\n"
    "frames = []\n"
    "if not df_kaggle.empty: frames.append(normalize(df_kaggle, 'title', 'text',    'date',        'kaggle'))\n"
    "if not df_hf.empty:     frames.append(normalize(df_hf,     'title', 'text',    'date',        'huggingface'))\n"
    "if not df_mongo.empty:  frames.append(normalize(df_mongo,  'title', 'content', 'publishedAt', 'newsapi'))\n"
    "\n"
    "df_final = pd.concat(frames, ignore_index=True)\n"
    "df_final.dropna(subset=['title', 'content_clean'], inplace=True)\n"
    "df_final.reset_index(drop=True, inplace=True)\n"
    "print(f'[Glue] Total filas unificadas: {len(df_final)}')\n"
    "\n"
    "# ── 5. Escribir en RDS ───────────────────────────────────────────────\n"
    "print('[Glue] Escribiendo en RDS...')\n"
    "conn = pymysql.connect(host=RDS_HOST, port=RDS_PORT, user=RDS_USER,\n"
    "                       password=RDS_PASSWORD, database=RDS_DB, charset='utf8mb4')\n"
    "with conn.cursor() as cur:\n"
    "    cur.execute(f'''\n"
    "        CREATE TABLE IF NOT EXISTS `{RDS_TABLE}` (\n"
    "            id            INT AUTO_INCREMENT PRIMARY KEY,\n"
    "            source        VARCHAR(50),\n"
    "            title         TEXT,\n"
    "            content_clean TEXT,\n"
    "            published_at  VARCHAR(50),\n"
    "            label         VARCHAR(20),\n"
    "            ingested_at   DATETIME DEFAULT CURRENT_TIMESTAMP\n"
    "        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n"
    "    ''')\n"
    "    rows = [\n"
    "        (str(r['source'])[:50], str(r['title'])[:500] if r['title'] else None,\n"
    "         str(r['content_clean'])[:5000] if r['content_clean'] else None,\n"
    "         str(r['published_at'])[:50],\n"
    "         str(r['label'])[:20] if r['label'] and str(r['label']) != 'None' else None)\n"
    "        for _, r in df_final.iterrows()\n"
    "    ]\n"
    "    cur.executemany(\n"
    "        f'INSERT INTO `{RDS_TABLE}` (source,title,content_clean,published_at,label) VALUES (%s,%s,%s,%s,%s)',\n"
    "        rows\n"
    "    )\n"
    "conn.commit()\n"
    "conn.close()\n"
    "print(f'[Glue] {len(rows)} filas escritas en RDS.')\n"
)


def upload_glue_script_to_s3():
    """Sube el script del job de Glue a S3."""
    log.info("── ETAPA 4A: Subiendo script de Glue a S3 ──")
    s3 = get_s3_client()
    s3_key = "scripts/glue_etl_job.py"
    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=GLUE_JOB_SCRIPT.encode("utf-8"),
        ContentType="text/x-python",
    )
    log.info(f"  ✓ Script subido → s3://{S3_BUCKET_NAME}/{s3_key}")


def create_or_update_glue_job():
    """
    Crea o actualiza el job de AWS Glue.
    Tipo pythonshell (sin Spark) — el más barato y rápido en Academy.
    MaxCapacity = 0.0625 DPU es el mínimo permitido.
    """
    log.info("── ETAPA 4B: Creando job en AWS Glue ──")
    glue = get_glue_client()

    job_config = {
        "Name": GLUE_JOB_NAME,
        "Role": GLUE_IAM_ROLE,
        "Command": {
            "Name":           "pythonshell",
            "ScriptLocation": GLUE_SCRIPT_S3_PATH,
            "PythonVersion":  "3",
        },
        "DefaultArguments": {
            "--TempDir":      GLUE_TEMP_DIR,
            "--job-language": "python",
            "--additional-python-modules": "pymysql,pymongo,pyarrow",
        },
        "GlueVersion":   "3.0",
        "MaxCapacity":   0.0625,     # 1/16 DPU — mínimo para pythonshell
        "Timeout":       60,         # minutos máximos
        "Description":   "ETL Hito1: S3 + MongoDB → limpieza → RDS",
    }

    try:
        glue.get_job(JobName=GLUE_JOB_NAME)
        update_cfg = {k: v for k, v in job_config.items() if k != "Name"}
        glue.update_job(JobName=GLUE_JOB_NAME, JobUpdate=update_cfg)
        log.info(f"  Job '{GLUE_JOB_NAME}' actualizado.")
    except glue.exceptions.EntityNotFoundException:
        glue.create_job(**job_config)
        log.info(f"  ✓ Job '{GLUE_JOB_NAME}' creado.")


def start_glue_job_and_wait():
    """
    Lanza el job de Glue y espera hasta que termine (polling cada 30 s).
    Estados finales: SUCCEEDED | FAILED | STOPPED | TIMEOUT | ERROR
    """
    log.info("── ETAPA 4C: Lanzando job de Glue ──")
    glue = get_glue_client()

    run_id = glue.start_job_run(JobName=GLUE_JOB_NAME)["JobRunId"]
    log.info(f"  Job run iniciado: {run_id}")

    while True:
        time.sleep(30)
        state = glue.get_job_run(JobName=GLUE_JOB_NAME, RunId=run_id)["JobRun"]["JobRunState"]
        log.info(f"  Estado Glue: {state}")
        if state == "SUCCEEDED":
            log.info("  ✓ Job de Glue completado con éxito.")
            break
        elif state in ("FAILED", "STOPPED", "TIMEOUT", "ERROR"):
            info = glue.get_job_run(JobName=GLUE_JOB_NAME, RunId=run_id)["JobRun"]
            log.error(f"  ✗ Job fallido [{state}]: {info.get('ErrorMessage', 'Sin detalles')}")
            break


# =============================================================================
# SECCIÓN 9 — ORQUESTADOR PRINCIPAL
# =============================================================================

def main():
    """
    Flujo completo:

      Kaggle CSV            ──► S3 (raw/kaggle/)
      HuggingFace Parquet   ──► S3 (raw/huggingface/)
      NewsAPI               ──► Kafka ──► MongoDB Atlas
      S3 + MongoDB          ──► AWS Glue ETL ──► RDS (raw_news_combined)
    """
    log.info("=" * 60)
    log.info("  INICIO DEL PIPELINE — HITO 1")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    if RUN_SETUP_S3:
        setup_s3_bucket()

    if RUN_INGEST_KAGGLE:
        ingest_kaggle_to_s3()

    if RUN_INGEST_HF:
        ingest_huggingface_to_s3()

    articles = []
    if RUN_INGEST_NEWSAPI:
        articles = fetch_newsapi_articles()

    if RUN_KAFKA_PRODUCER and articles:
        produce_to_kafka(articles)

    if RUN_INGEST_MONGO and articles:
        save_articles_to_mongo(articles)

    if RUN_GLUE_UPLOAD:
        upload_glue_script_to_s3()

    if RUN_GLUE_CREATE_JOB:
        create_or_update_glue_job()

    if RUN_GLUE_START_JOB:
        start_glue_job_and_wait()

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETADO")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
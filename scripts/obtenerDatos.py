# =============================================================================
# EXTRACCIÓN DE DATOS DESDE RDS PARA EDA — LOCAL
# =============================================================================
# Descarga los datos unificados desde MariaDB RDS a un archivo CSV local
# utilizando transporte cifrado seguro (SSL), dejándolo listo para el EDA.
# =============================================================================

# =============================================================================
# SECCIÓN 1 — IMPORTS
# =============================================================================
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Cargar las variables del archivo .env localizado en la raíz del proyecto
load_dotenv()

# =============================================================================
# SECCIÓN 2 — VARIABLES DE CONFIGURACIÓN
# =============================================================================
RDS_HOST      = os.getenv("RDS_HOST")
RDS_PORT      = int(os.getenv("RDS_PORT", 3306))
RDS_DB_NAME   = os.getenv("RDS_DB_NAME", "noticias_hito1")
RDS_USER      = os.getenv("RDS_USER", "admin")
RDS_PASSWORD  = os.getenv("RDS_PASSWORD")
RDS_TABLE     = os.getenv("RDS_TABLE", "raw_news_combined")

# Ruta de destino local para almacenar el CSV generado
OUTPUT_CSV    = "./datasets/eda/raw_news_combined.csv"

# =============================================================================
# SECCIÓN 3 — CONEXIÓN Y DESCARGA ENCRIPTADA (SSL)
# =============================================================================

def get_db_engine():
    """Genera el motor de conexión optimizado con SSL obligatorio."""
    # Cadena de conexión estándar para protocolo MariaDB/MySQL
    connection_uri = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
    
    # Configuramos el motor inyectando el SSL para cumplir con --require_secure_transport=ON
    return create_engine(
        connection_uri,
        connect_args={
            "charset": "utf8mb4",
            "ssl": {"ssl": True}  # Resuelve el error de transporte inseguro (Error 3159)
        }
    )

def download_all_data() -> pd.DataFrame:
    """Se conecta de forma segura a MariaDB, descarga la tabla y muestra métricas."""
    print("Connecting to MariaDB RDS (via Secure Transport/SSL)...")
    engine = get_db_engine()
    
    print(f"Downloading data from table '{RDS_TABLE}'...")
    # Selección explícita de columnas del esquema unificado
    query = f"SELECT id, source, title, content_clean, published_at, label, ingested_at FROM `{RDS_TABLE}`"
    
    # Pandas procesa la consulta y genera el DataFrame
    df = pd.read_sql(query, con=engine)
    
    print(f"✓ {len(df)} rows successfully downloaded.")
    
    # Métricas de validación por consola
    print(f"\n[Control] Distribution by Source:")
    print(df["source"].value_counts().to_string())
    
    print(f"\n[Control] Distribution by Target Label:")
    print(df["label"].value_counts(dropna=False).to_string())
    
    return df

def save_to_csv(df: pd.DataFrame):
    """Guarda el DataFrame en local asegurando la integridad del texto."""
    # Crear el directorio base en la ruta especificada si no existe
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Forzamos la codificación UTF-8 para proteger tildes y caracteres especiales del castellano
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓ CSV dataset successfully saved to: {OUTPUT_CSV}")

# =============================================================================
# SECCIÓN 4 — EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    try:
        # Ejecución del pipeline de volcado local
        df_news = download_all_data()
        save_to_csv(df_news)
        
        print("\n=== PREVIEW: First 5 Records ===")
        print(df_news.head(5))
        
        print("\n=== METADATA: Dataframe Structural Info ===")
        print(df_news.info())
        
    except Exception as e:
        print(f"\n✗ Critical Error during data extraction: {e}")
        print("Please check your .env credentials, your internet connection, or your RDS Security Group rules.")
# core/db_connect.py
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("DB_HOST")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{dbname}"
engine = create_engine(conn_str)

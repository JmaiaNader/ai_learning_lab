from contextlib import contextmanager
import mysql.connector
from config.settings import settings

@contextmanager
def mysql_connection():
    conn = mysql.connector.connect(host=settings.mysql_host, port=settings.mysql_port,
        user=settings.mysql_user, password=settings.mysql_password,
        database=settings.mysql_database, connection_timeout=5)
    try:
        yield conn
    finally:
        conn.close()

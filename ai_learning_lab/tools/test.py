from dotenv import load_dotenv
import os

load_dotenv()

print("MYSQL_HOST:", os.getenv("MYSQL_HOST"))
print("MYSQL_DATABASE:", os.getenv("MYSQL_DATABASE"))
print("MYSQL_USER:", os.getenv("MYSQL_USER"))
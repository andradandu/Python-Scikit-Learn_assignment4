import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
    
    # Format: postgresql://username@localhost:5432/database_name
    # Note: No colon after the username because there is no password
    SQLALCHEMY_DATABASE_URI = "postgresql://andradandu@localhost:5432/ml_app"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
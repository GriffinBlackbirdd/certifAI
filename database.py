from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent


STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)


DB_FILE = STATIC_DIR / "course_content.db"


DATABASE_URL = f"sqlite:///{DB_FILE}"


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


def init_db():
    from . import models
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
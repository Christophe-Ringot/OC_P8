from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .db_models import Base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./logs.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Mettre à True pour debug SQL
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# Production Engine Configuration
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=20,          # Maintain 20 active connections in the pool
    max_overflow=10,       # Allow up to 10 extra connections during traffic spikes
    pool_pre_ping=True,    # Verify connection is alive before using it (prevents 500 errors)
    pool_timeout=30        # Wait 30 seconds for a connection before failing
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
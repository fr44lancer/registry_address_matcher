import streamlit as st
import logging
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@st.cache_resource
def get_database_connection(host="localhost", port=3306, database="experiments", user="root", password="DC123456"):
    """Create database connection with error handling"""
    try:
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        logger.error(f"Database connection error: {str(e)}")
        return None
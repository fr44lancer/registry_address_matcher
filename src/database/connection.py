import logging
from typing import Optional
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.engine import Connection
import streamlit as st

from config.settings import DB_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles database connections and operations"""
    
    def __init__(self, config=None):
        self.config = config or DB_CONFIG
        self._engine: Optional[Engine] = None
    
    @property
    def engine(self) -> Optional[Engine]:
        """Get database engine, creating it if necessary"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    def _create_engine(self) -> Optional[Engine]:
        """Create database engine with error handling"""
        try:
            connection_string = (
                f"mysql+pymysql://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
                f"?charset=utf8mb4"
            )
            
            engine = create_engine(
                connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,
                echo=False
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established successfully")
            return engine
            
        except Exception as e:
            error_msg = f"Database connection failed: {str(e)}"
            logger.error(error_msg)
            if 'st' in globals():
                st.error(error_msg)
            return None
    
    def get_connection(self) -> Optional[Connection]:
        """Get a database connection"""
        if self.engine is None:
            return None
        try:
            return self.engine.connect()
        except Exception as e:
            logger.error(f"Failed to get database connection: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                if conn:
                    conn.execute(text("SELECT 1"))
                    return True
            return False
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connections closed")


# Global database manager instance
@st.cache_resource
def get_database_manager() -> DatabaseManager:
    """Get cached database manager instance"""
    return DatabaseManager()
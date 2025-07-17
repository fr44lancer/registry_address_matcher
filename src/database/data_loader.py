import logging
import pandas as pd
from typing import Optional, List
import streamlit as st
from sqlalchemy import Engine

from config.settings import REQUIRED_COLUMNS, CORE_ADDRESS_COLUMNS, APP_CONFIG
from .connection import DatabaseManager

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from database with validation and caching"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def load_registry_data(self, registry_name: str, table_name: str) -> Optional[pd.DataFrame]:
        """Load registry data with comprehensive error handling and validation"""
        try:
            engine = self.db_manager.engine
            if not engine:
                st.error("Database connection not available")
                return None
            
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, con=engine)
            
            # Validate required columns
            if not self._validate_columns(df, registry_name):
                return None
            
            logger.info(f"Loaded {len(df)} records from {registry_name}")
            return df
            
        except Exception as e:
            error_msg = f"Error loading {registry_name} data: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            return None
    
    def _validate_columns(self, df: pd.DataFrame, registry_name: str) -> bool:
        """Validate that required columns exist in the dataframe"""
        registry_type = registry_name.lower()
        required_columns = REQUIRED_COLUMNS.get(registry_type, CORE_ADDRESS_COLUMNS)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns in {registry_name}: {missing_columns}"
            st.error(error_msg)
            logger.error(error_msg)
            return False
        
        return True
    
    def get_table_info(self, table_name: str) -> Optional[dict]:
        """Get information about a database table"""
        try:
            engine = self.db_manager.engine
            if not engine:
                return None
            
            # Get table schema
            query = f"DESCRIBE {table_name}"
            schema_df = pd.read_sql(query, con=engine)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_df = pd.read_sql(count_query, con=engine)
            
            return {
                'schema': schema_df.to_dict('records'),
                'row_count': count_df.iloc[0]['count']
            }
            
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return None
    
    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        try:
            engine = self.db_manager.engine
            if not engine:
                return []
            
            query = "SHOW TABLES"
            tables_df = pd.read_sql(query, con=engine)
            return tables_df.iloc[:, 0].tolist()
            
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []


@st.cache_data(ttl=APP_CONFIG.cache_ttl)
def load_registry_data_cached(registry_name: str, table_name: str, _db_manager: DatabaseManager) -> Optional[pd.DataFrame]:
    """Cached version of registry data loading"""
    data_loader = DataLoader(_db_manager)
    return data_loader.load_registry_data(registry_name, table_name)
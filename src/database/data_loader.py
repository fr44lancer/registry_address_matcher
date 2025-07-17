import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_registry_data(registry_name, table_name, _engine):
    """Load registry data with comprehensive error handling"""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, con=_engine)

        # Data validation
        required_columns = ['STREET_NAME', 'HOUSE', 'BUILDING']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns in {registry_name}: {missing_columns}")
            return None

        logger.info(f"Loaded {len(df)} records from {registry_name}")
        return df

    except Exception as e:
        st.error(f"Error loading {registry_name} data: {str(e)}")
        logger.error(f"Data loading error for {registry_name}: {str(e)}")
        return None
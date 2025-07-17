import pandas as pd
import streamlit as st
import logging
from ..matching.normalizer import AddressNormalizer

logger = logging.getLogger(__name__)


@st.cache_data
def preprocess_registries(_spr_df, _cad_df):
    """Comprehensive data preprocessing pipeline"""
    normalizer = AddressNormalizer()

    def process_registry(df, registry_name):
        """Process individual registry with comprehensive normalization"""
        processed = df.copy()

        # Handle missing values
        processed['STREET_NAME'] = processed['STREET_NAME'].fillna('')
        processed['HOUSE'] = processed['HOUSE'].fillna('')
        processed['BUILDING'] = processed['BUILDING'].fillna('')

        # Normalize fields
        processed['STREET_NORM'] = processed['STREET_NAME'].apply(normalizer.normalize)
        processed['HOUSE_NORM'] = processed['HOUSE'].apply(normalizer.normalize)
        processed['BUILDING_NORM'] = processed['BUILDING'].apply(normalizer.normalize)

        # Create composite addresses
        processed['FULL_ADDRESS'] = (
                processed['STREET_NORM'] + " " +
                processed['HOUSE_NORM'] + " " +
                processed['BUILDING_NORM']
        ).str.strip()

        # Create search keys
        processed['SEARCH_KEY'] = (
                processed['STREET_NORM'] + "_" + processed['HOUSE_NORM']
        )

        # Data quality metrics
        processed['COMPLETENESS_SCORE'] = (
                                                  processed['STREET_NAME'].notna().astype(int) +
                                                  processed['HOUSE'].notna().astype(int) +
                                                  processed['BUILDING'].notna().astype(int)
                                          ) / 3

        logger.info(f"Processed {len(processed)} records for {registry_name}")
        return processed

    spr_processed = process_registry(_spr_df, "SPR")
    cad_processed = process_registry(_cad_df, "Cadastre")

    return spr_processed, cad_processed
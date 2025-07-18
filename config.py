import os
import logging
from typing import Dict, Any


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('address_matching.log'),
            logging.StreamHandler()
        ]
    )


def get_database_config() -> Dict[str, Any]:
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'database': os.getenv('DB_NAME', 'experiments'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', 'DC123456')
    }


def get_app_config() -> Dict[str, Any]:
    """Get application configuration"""
    return {
        'page_title': "Address Registry Matcher",
        'page_icon': "🏘️",
        'layout': "wide",
        'initial_sidebar_state': "expanded",
        'default_spr_table': "spr",
        'default_cad_table': "cadastre_dp",
        'default_chunk_size': 500,
        'default_threshold': 85,
        'default_max_records': 10000,
        'cache_ttl': 3600,  # 1 hour
        'csv_paths': {
            'spr': "data/spr.csv",
            'cadastre': "data/cadastre.csv"
        }
    }


def get_matching_config() -> Dict[str, Any]:
    """Get matching algorithm configuration"""
    return {
        'fuzzy_strategies': [
            ('token_sort_ratio', 'Token Sort Ratio'),
            ('token_set_ratio', 'Token Set Ratio'),
            ('partial_ratio', 'Partial Ratio'),
            ('ratio', 'Simple Ratio')
        ],
        'score_thresholds': {
            'excellent': 95,
            'good': 85,
            'fair': 75,
            'poor': 0
        },
        'quality_categories': {
            'high': 90,
            'medium': 80,
            'low': 0
        }
    }


# Armenian language specific configuration
ARMENIAN_CONFIG = {
    'suffixes': [
        r'\bԽՃՂ\.?', r'\bՃՂ\.?', r'\bՓ\.?', r'\bՊՈՂ\.?', 
        r'\bԱՎ\.?', r'\bՃԱՄԲ\.?', r'\bԹԵԼԱ\.?'
    ],
    'aliases': {
        "Խ. ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
        "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
    }
}


# Export configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'xlsx', 'json', 'zip'],
    'timestamp_format': '%Y%m%d_%H%M%S',
    'zip_compression': 'ZIP_DEFLATED',
    'csv_encoding': 'utf-8-sig',
    'excel_engine': 'openpyxl'
}
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 3306
    database: str = "experiments"
    user: str = "root"
    password: str = "DC123456"
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle: int = 3600


@dataclass
class MatchingConfig:
    threshold_excellent: float = 90.0
    threshold_good: float = 75.0
    threshold_poor: float = 50.0
    max_results: int = 100
    fuzzy_ratio_weight: float = 0.4
    partial_ratio_weight: float = 0.3
    token_sort_weight: float = 0.2
    token_set_weight: float = 0.1


@dataclass
class AppConfig:
    page_title: str = "Address Registry Matcher"
    page_icon: str = "🏘️"
    layout: str = "wide"
    cache_ttl: int = 3600
    log_level: str = "INFO"
    log_file: str = "address_matching.log"


# Required columns for each registry type (from original address.py)
REQUIRED_COLUMNS = {
    'spr': ['STREET_NAME', 'HOUSE', 'BUILDING'],
    'cad': ['STREET_NAME', 'HOUSE', 'BUILDING']
}

# Core address columns used in processing
CORE_ADDRESS_COLUMNS = ['STREET_NAME', 'HOUSE', 'BUILDING']

# Generated columns during processing
GENERATED_COLUMNS = {
    'STREET_NORM': 'Normalized street name',
    'HOUSE_NORM': 'Normalized house number', 
    'BUILDING_NORM': 'Normalized building number',
    'FULL_ADDRESS': 'Full concatenated address',
    'SEARCH_KEY': 'Search key for matching',
    'COMPLETENESS_SCORE': 'Data completeness score (0-1)'
}

# Application settings
APP_CONFIG = AppConfig()
DB_CONFIG = DatabaseConfig()
MATCHING_CONFIG = MatchingConfig()
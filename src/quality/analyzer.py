def analyze_data_quality(df, registry_name):
    """Comprehensive data quality analysis"""
    quality_metrics = {
        'total_records': len(df),
        'street_completeness': df['STREET_NAME'].notna().sum() / len(df),
        'house_completeness': df['HOUSE'].notna().sum() / len(df),
        'building_completeness': df['BUILDING'].notna().sum() / len(df),
        'unique_streets': df['STREET_NORM'].nunique(),
        'avg_completeness': df['COMPLETENESS_SCORE'].mean(),
        'duplicate_addresses': len(df) - df['FULL_ADDRESS'].nunique(),
    }

    return quality_metrics
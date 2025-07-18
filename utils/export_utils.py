import pandas as pd
import json
import zipfile
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional


def create_export_package(matches_df: pd.DataFrame, spr_df: pd.DataFrame, cad_df: pd.DataFrame, quality_metrics: Dict[str, Any]) -> bytes:
    """Create comprehensive export package"""
    
    # Create zip buffer
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add matched results
        if not matches_df.empty:
            matches_csv = matches_df.to_csv(index=False)
            zip_file.writestr('matched_addresses.csv', matches_csv)
        
        # Add unmatched records
        if not matches_df.empty and not spr_df.empty:
            matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique())
            unmatched_spr = spr_df[~spr_df.get('ADDRESS_ID', spr_df.index).isin(matched_spr_ids)]
            if not unmatched_spr.empty:
                unmatched_csv = unmatched_spr.to_csv(index=False)
                zip_file.writestr('unmatched_spr_addresses.csv', unmatched_csv)
        
        # Add quality report
        if quality_metrics:
            quality_report = json.dumps(quality_metrics, indent=2, default=str)
            zip_file.writestr('quality_report.json', quality_report)
        
        # Add matching statistics
        if not matches_df.empty:
            stats = {
                'total_spr_records': len(spr_df) if not spr_df.empty else 0,
                'total_cad_records': len(cad_df) if not cad_df.empty else 0,
                'total_matches': len(matches_df),
                'match_rate': len(matches_df) / len(spr_df) if not spr_df.empty and len(spr_df) > 0 else 0,
                'match_types': matches_df['MATCH_TYPE'].value_counts().to_dict() if 'MATCH_TYPE' in matches_df.columns else {},
                'score_statistics': {
                    'mean': float(matches_df['MATCH_SCORE'].mean()) if 'MATCH_SCORE' in matches_df.columns else 0,
                    'median': float(matches_df['MATCH_SCORE'].median()) if 'MATCH_SCORE' in matches_df.columns else 0,
                    'std': float(matches_df['MATCH_SCORE'].std()) if 'MATCH_SCORE' in matches_df.columns else 0,
                    'min': float(matches_df['MATCH_SCORE'].min()) if 'MATCH_SCORE' in matches_df.columns else 0,
                    'max': float(matches_df['MATCH_SCORE'].max()) if 'MATCH_SCORE' in matches_df.columns else 0
                }
            }
            zip_file.writestr('matching_statistics.json', json.dumps(stats, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_matches_to_csv(matches_df: pd.DataFrame) -> str:
    """Export matches to CSV format"""
    return matches_df.to_csv(index=False)


def export_unmatched_to_csv(spr_df: pd.DataFrame, matches_df: pd.DataFrame) -> str:
    """Export unmatched records to CSV format"""
    if matches_df.empty:
        return spr_df.to_csv(index=False)
    
    matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique())
    unmatched_spr = spr_df[~spr_df.get('ADDRESS_ID', spr_df.index).isin(matched_spr_ids)]
    return unmatched_spr.to_csv(index=False)


def create_summary_report(matches_df: pd.DataFrame, quality_metrics: Dict[str, Any], spr_df: pd.DataFrame) -> str:
    """Create a markdown summary report"""
    
    # Calculate key metrics
    total_spr = len(spr_df) if not spr_df.empty else 0
    total_matches = len(matches_df)
    match_rate = total_matches / total_spr if total_spr > 0 else 0
    
    avg_score = matches_df['MATCH_SCORE'].mean() if not matches_df.empty and 'MATCH_SCORE' in matches_df.columns else 0
    avg_score_text = f"{avg_score:.1f}" if not matches_df.empty else "N/A"
    
    high_quality_count = len(matches_df[matches_df['MATCH_SCORE'] >= 90]) if not matches_df.empty and 'MATCH_SCORE' in matches_df.columns else 0
    medium_quality_count = len(matches_df[(matches_df['MATCH_SCORE'] >= 80) & (matches_df['MATCH_SCORE'] < 90)]) if not matches_df.empty and 'MATCH_SCORE' in matches_df.columns else 0
    low_quality_count = len(matches_df[matches_df['MATCH_SCORE'] < 80]) if not matches_df.empty and 'MATCH_SCORE' in matches_df.columns else 0
    
    # Generate recommendations
    recommendations = []
    if not matches_df.empty:
        if avg_score < 85:
            recommendations.append("Consider lowering the matching threshold to capture more potential matches")
        
        if low_quality_count > total_matches * 0.2:
            recommendations.append("High number of low-quality matches - review and possibly adjust matching parameters")
    
    if match_rate < 0.5:
        recommendations.append("Low match rate - consider data quality improvements or relaxed matching criteria")
    
    summary_text = f"""
# Address Matching Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method:** {quality_metrics.get('matching_method', 'Unknown') if quality_metrics else 'Unknown'}
**Processing Time:** {quality_metrics.get('processing_time', 0):.2f} seconds

## Results Overview
- **Total SPR Records:** {total_spr:,}
- **Total Matches:** {total_matches:,}
- **Match Rate:** {match_rate:.1%}
- **Average Score:** {avg_score_text}

## Quality Assessment
- **High Quality Matches (≥90):** {high_quality_count}
- **Medium Quality Matches (80-89):** {medium_quality_count}
- **Low Quality Matches (<80):** {low_quality_count}

## Recommendations
{chr(10).join(f"- {rec}" for rec in recommendations) if recommendations else "- No specific recommendations"}
"""
    
    return summary_text


def create_quality_report(spr_quality: Dict[str, Any], cad_quality: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive quality report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'spr_registry': {
            'total_records': spr_quality.get('total_records', 0),
            'completeness': {
                'street_completeness': spr_quality.get('street_completeness', 0),
                'house_completeness': spr_quality.get('house_completeness', 0),
                'building_completeness': spr_quality.get('building_completeness', 0),
                'average_completeness': spr_quality.get('avg_completeness', 0)
            },
            'data_quality': {
                'unique_streets': spr_quality.get('unique_streets', 0),
                'duplicate_addresses': spr_quality.get('duplicate_addresses', 0)
            }
        },
        'cadastre_registry': {
            'total_records': cad_quality.get('total_records', 0),
            'completeness': {
                'street_completeness': cad_quality.get('street_completeness', 0),
                'house_completeness': cad_quality.get('house_completeness', 0),
                'building_completeness': cad_quality.get('building_completeness', 0),
                'average_completeness': cad_quality.get('avg_completeness', 0)
            },
            'data_quality': {
                'unique_streets': cad_quality.get('unique_streets', 0),
                'duplicate_addresses': cad_quality.get('duplicate_addresses', 0)
            }
        }
    }
    
    return report


def export_to_excel(matches_df: pd.DataFrame, unmatched_df: Optional[pd.DataFrame] = None, filename: str = "address_matching_results.xlsx") -> BytesIO:
    """Export results to Excel format with multiple sheets"""
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Export matched results
        if not matches_df.empty:
            matches_df.to_excel(writer, sheet_name='Matched_Addresses', index=False)
        
        # Export unmatched results
        if unmatched_df is not None and not unmatched_df.empty:
            unmatched_df.to_excel(writer, sheet_name='Unmatched_Addresses', index=False)
        
        # Create summary sheet
        if not matches_df.empty:
            summary_data = {
                'Metric': ['Total Matches', 'Average Score', 'High Quality (≥90)', 'Medium Quality (80-89)', 'Low Quality (<80)'],
                'Value': [
                    len(matches_df),
                    matches_df['MATCH_SCORE'].mean() if 'MATCH_SCORE' in matches_df.columns else 0,
                    len(matches_df[matches_df['MATCH_SCORE'] >= 90]) if 'MATCH_SCORE' in matches_df.columns else 0,
                    len(matches_df[(matches_df['MATCH_SCORE'] >= 80) & (matches_df['MATCH_SCORE'] < 90)]) if 'MATCH_SCORE' in matches_df.columns else 0,
                    len(matches_df[matches_df['MATCH_SCORE'] < 80]) if 'MATCH_SCORE' in matches_df.columns else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output
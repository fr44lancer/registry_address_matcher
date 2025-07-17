import json
from io import BytesIO
import zipfile


def create_export_package(matches_df, spr_df, cad_df, quality_metrics):
    """Create comprehensive export package"""

    # Create zip buffer
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add matched results
        matches_csv = matches_df.to_csv(index=False)
        zip_file.writestr('matched_addresses.csv', matches_csv)

        # Add unmatched records
        matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique())
        unmatched_spr = spr_df[~spr_df.get('ADDRESS_ID', spr_df.index).isin(matched_spr_ids)]
        unmatched_csv = unmatched_spr.to_csv(index=False)
        zip_file.writestr('unmatched_spr_addresses.csv', unmatched_csv)

        # Add quality report
        quality_report = json.dumps(quality_metrics, indent=2)
        zip_file.writestr('quality_report.json', quality_report)

        # Add matching statistics
        stats = {
            'total_spr_records': len(spr_df),
            'total_cad_records': len(cad_df),
            'total_matches': len(matches_df),
            'match_rate': len(matches_df) / len(spr_df) if len(spr_df) > 0 else 0,
            'match_types': matches_df['MATCH_TYPE'].value_counts().to_dict(),
            'score_statistics': {
                'mean': matches_df['MATCH_SCORE'].mean(),
                'median': matches_df['MATCH_SCORE'].median(),
                'std': matches_df['MATCH_SCORE'].std(),
                'min': matches_df['MATCH_SCORE'].min(),
                'max': matches_df['MATCH_SCORE'].max()
            }
        }
        zip_file.writestr('matching_statistics.json', json.dumps(stats, indent=2))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
import zipfile
import json


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Address Registry Matcher",
        page_icon="🏘️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #2a5298;
        }
        .match-quality-excellent { background-color: #d4edda; }
        .match-quality-good { background-color: #fff3cd; }
        .match-quality-poor { background-color: #f8d7da; }
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        }
    </style>
    """, unsafe_allow_html=True)


def render_main_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏘️ Address Matcher</h1>
        <p>Comprehensive address matching and mapping between SPR and Cadastre registries</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_config():
    """Render the sidebar configuration section"""
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Configuration")

        # Database settings
        import os
        default_host = os.getenv('DB_HOST', 'localhost')
        default_port = int(os.getenv('DB_PORT', '3306'))
        default_db = os.getenv('DB_NAME', 'experiments')
        default_user = os.getenv('DB_USER', 'root')
        default_password = os.getenv('DB_PASSWORD', 'DC123456')

        db_config = {
            'host': default_host,
            'port': default_port,
            'database': default_db,
            'user': default_user,
            'password': default_password
        }

        # Table settings
        table_config = {
            'spr_table': "spr",
            'cad_table': "cadastre_dp"
        }

        # Matching parameters - fixed to exact matching only
        st.info("🎯 **Matching Strategy:** Exact matching only - fast and precise address matching")

        # Processing limits
        st.subheader("Processing Configuration")
        col1, col2 = st.columns(2)

        with col1:
            use_all_records = st.checkbox(
                "Process All Records",
                value=True,
                help="Process all available SPR records (ignores Max Records limit)"
            )

            if not use_all_records:
                max_records = st.number_input(
                    "Max Records to Process",
                    min_value=1,
                    max_value=1000000,
                    value=10000,
                    step=100,
                    help="Total number of SPR records to process for matching"
                )
            else:
                max_records = None

        with col2:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=10,
                max_value=10000,
                value=500,
                step=50,
                help="Number of records to process in each chunk (affects memory usage and progress updates)"
            )


        # Advanced options
        st.subheader("Advanced Options")
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

        return {
            'db_config': db_config,
            'table_config': table_config,
            'matching_method': 'Exact Only',
            'use_all_records': use_all_records,
            'max_records': max_records,
            'chunk_size': chunk_size,
            'enable_logging': enable_logging
        }


def render_data_overview_metrics(spr_quality, cad_quality):
    """Render data overview metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("SPR Records", f"{spr_quality['total_records']:,}")
        st.metric("Unique Streets", f"{spr_quality['unique_streets']:,}")

    with col2:
        st.metric("Cadastre Records", f"{cad_quality['total_records']:,}")
        st.metric("Unique Streets", f"{cad_quality['unique_streets']:,}")

    with col3:
        st.metric("SPR Completeness", f"{spr_quality['avg_completeness']:.1%}")
        st.metric("Duplicates", f"{spr_quality['duplicate_addresses']:,}")

    with col4:
        st.metric("Cadastre Completeness", f"{cad_quality['avg_completeness']:.1%}")
        st.metric("Duplicates", f"{cad_quality['duplicate_addresses']:,}")


def render_sample_data_preview(spr_processed, cad_processed):
    """Render sample data preview"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SPR Sample Data")
        st.dataframe(
            spr_processed[['STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))

    with col2:
        st.subheader("Cadastre Sample Data")
        st.dataframe(
            cad_processed[['STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))


def render_matching_controls():
    """Render matching control buttons"""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        start_button = st.button("🚀 Start Matching Process", type="primary", use_container_width=True)

    with col2:
        stop_button = st.button("⏹ Stop Process", use_container_width=True)

    with col3:
        reset_button = st.button("🔄 Reset", use_container_width=True)

    return start_button, stop_button, reset_button


def render_matching_results_summary(matches_df, quality_metrics):
    """Render matching results summary"""
    processing_records = quality_metrics.get('processing_records', 0)
    total_records = quality_metrics.get('total_spr_records', 0)
    coverage_pct = quality_metrics.get('records_processed_percentage', 100)

    # Quick statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(matches_df))
    
    with col2:
        match_rate = len(matches_df) / processing_records * 100 if processing_records > 0 else 0
        st.metric("Match Rate", f"{match_rate:.1f}%")
    
    with col3:
        avg_score = matches_df['MATCH_SCORE'].mean() if len(matches_df) > 0 else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    with col4:
        processing_time = st.session_state.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")

    # Processing coverage info
    if processing_records < total_records:
        st.info(
            f"📊 **Processing Coverage:** {processing_records:,} of {total_records:,} total records ({coverage_pct:.1f}%)")
    else:
        st.info(f"📊 **Processing Coverage:** All {total_records:,} records processed (100%)")


def render_match_type_distribution(matches_df):
    """Render match type distribution"""
    if len(matches_df) > 0:
        match_type_counts = matches_df['MATCH_TYPE'].value_counts()
        st.subheader("Match Type Distribution")

        # Create columns for each match type
        cols = st.columns(len(match_type_counts))
        for i, (match_type, count) in enumerate(match_type_counts.items()):
            with cols[i]:
                percentage = count / len(matches_df) * 100
                st.metric(match_type, f"{count} ({percentage:.1f}%)")


def render_score_quality_categories(matches_df):
    """Render score quality categories"""
    st.subheader("Score Quality Categories")

    # Categorize matches by score
    excellent = matches_df[matches_df['MATCH_SCORE'] >= 95]
    good = matches_df[(matches_df['MATCH_SCORE'] >= 85) & (matches_df['MATCH_SCORE'] < 95)]
    fair = matches_df[(matches_df['MATCH_SCORE'] >= 75) & (matches_df['MATCH_SCORE'] < 85)]
    poor = matches_df[matches_df['MATCH_SCORE'] < 75]

    st.success(f"🟢 Excellent (95-100): {len(excellent)} matches")
    st.info(f"🔵 Good (85-94): {len(good)} matches")
    st.warning(f"🟡 Fair (75-84): {len(fair)} matches")
    st.error(f"🔴 Poor (<75): {len(poor)} matches")


def render_top_scoring_matches(matches_df):
    """Render top scoring matches"""
    st.subheader("Top Scoring Matches")
    top_matches = matches_df.nlargest(10, 'MATCH_SCORE')[
        ['STREET_NAME_SPR', 'HOUSE_SPR', 'STREET_NAME_CAD', 'HOUSE_CAD', 'MATCH_SCORE']
    ]
    st.dataframe(top_matches, use_container_width=True)


def render_interactive_match_explorer(matches_df):
    """Render interactive match explorer"""
    st.subheader("Interactive Match Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        score_filter = st.slider("Minimum Score", 0, 100, 70)

    with col2:
        match_type_filter = st.multiselect(
            "Match Types",
            options=matches_df['MATCH_TYPE'].unique(),
            default=matches_df['MATCH_TYPE'].unique()
        )

    with col3:
        street_filter = st.text_input("Filter by Street Name", "")

    # Apply filters
    filtered_matches = matches_df[
        (matches_df['MATCH_SCORE'] >= score_filter) &
        (matches_df['MATCH_TYPE'].isin(match_type_filter))
    ]

    if street_filter:
        filtered_matches = filtered_matches[
            filtered_matches['STREET_NAME_SPR'].str.contains(street_filter, case=False, na=False) |
            filtered_matches['STREET_NAME_CAD'].str.contains(street_filter, case=False, na=False)
        ]

    # Display filtered results
    st.subheader(f"Filtered Results ({len(filtered_matches)} matches)")

    # Pagination
    page_size = 50
    total_pages = (len(filtered_matches) - 1) // page_size + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

    start_idx = page * page_size
    end_idx = start_idx + page_size

    display_matches = filtered_matches.iloc[start_idx:end_idx]

    # Enhanced display with color coding
    def highlight_score(val):
        if val >= 95:
            return 'background-color: #d4edda'
        elif val >= 85:
            return 'background-color: #d1ecf1'
        elif val >= 75:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'

    styled_matches = display_matches.style.applymap(
        highlight_score, subset=['MATCH_SCORE']
    )

    st.dataframe(styled_matches, use_container_width=True)

    return filtered_matches


def render_manual_review_interface(filtered_matches):
    """Render manual review interface"""
    st.subheader("Manual Review Interface")

    if len(filtered_matches) > 0:
        review_idx = st.selectbox(
            "Select match to review",
            range(len(filtered_matches)),
            format_func=lambda x: f"Match {x + 1}: {filtered_matches.iloc[x]['STREET_NAME_SPR']} → {filtered_matches.iloc[x]['STREET_NAME_CAD']} (Score: {filtered_matches.iloc[x]['MATCH_SCORE']})"
        )

        if review_idx is not None:
            review_match = filtered_matches.iloc[review_idx]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("SPR Record")
                st.write(f"**Street:** {review_match['STREET_NAME_SPR']}")
                st.write(f"**House:** {review_match['HOUSE_SPR']}")
                st.write(f"**Building:** {review_match['BUILDING_SPR']}")
                st.write(f"**Full Address:** {review_match['FULL_ADDRESS_SPR']}")

            with col2:
                st.subheader("Cadastre Record")
                st.write(f"**Street:** {review_match['STREET_NAME_CAD']}")
                st.write(f"**House:** {review_match['HOUSE_CAD']}")
                st.write(f"**Building:** {review_match['BUILDING_CAD']}")
                st.write(f"**Full Address:** {review_match['FULL_ADDRESS_CAD']}")

            # Match details
            st.subheader("Match Details")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Match Score", f"{review_match['MATCH_SCORE']:.1f}")

            with col2:
                st.metric("Match Type", review_match['MATCH_TYPE'])

            with col3:
                st.metric("Candidates Evaluated", review_match.get('CANDIDATES_COUNT', 'N/A'))

            # Manual confirmation
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("✅ Confirm Match", key=f"confirm_{review_idx}"):
                    st.success("Match confirmed!")

            with col2:
                if st.button("❌ Reject Match", key=f"reject_{review_idx}"):
                    st.error("Match rejected!")

            with col3:
                if st.button("❓ Mark for Review", key=f"review_{review_idx}"):
                    st.warning("Marked for further review!")


def render_quality_report_summary(matches_df, quality_metrics, spr_processed):
    """Render quality report summary"""
    st.subheader("📊 Matching Performance Report")

    # Executive summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Executive Summary")
        total_spr = len(spr_processed)
        total_matches = len(matches_df)
        match_rate = total_matches / total_spr if total_spr > 0 else 0

        st.write(f"**Total SPR Records:** {total_spr:,}")
        st.write(f"**Total Matches Found:** {total_matches:,}")
        st.write(f"**Overall Match Rate:** {match_rate:.1%}")
        st.write(f"**Processing Time:** {quality_metrics.get('processing_time', 0):.2f} seconds")
        st.write(f"**Matching Method:** {quality_metrics.get('matching_method', 'Unknown')}")

    with col2:
        st.markdown("### Quality Indicators")
        if len(matches_df) > 0:
            avg_score = matches_df['MATCH_SCORE'].mean()
            high_quality = len(matches_df[matches_df['MATCH_SCORE'] >= 90])
            low_quality = len(matches_df[matches_df['MATCH_SCORE'] < 80])

            st.write(f"**Average Match Score:** {avg_score:.1f}")
            st.write(f"**High Quality Matches (≥90):** {high_quality} ({high_quality / len(matches_df) * 100:.1f}%)")
            st.write(f"**Low Quality Matches (<80):** {low_quality} ({low_quality / len(matches_df) * 100:.1f}%)")

            # Quality assessment
            if avg_score >= 90:
                st.success("🟢 Excellent matching quality")
            elif avg_score >= 80:
                st.info("🔵 Good matching quality")
            elif avg_score >= 70:
                st.warning("🟡 Fair matching quality")
            else:
                st.error("🔴 Poor matching quality - review parameters")


def render_detailed_statistics(matches_df):
    """Render detailed statistics"""
    st.subheader("📈 Detailed Statistics")

    if len(matches_df) > 0:
        # Score statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Score Distribution")
            st.write(f"**Mean:** {matches_df['MATCH_SCORE'].mean():.2f}")
            st.write(f"**Median:** {matches_df['MATCH_SCORE'].median():.2f}")
            st.write(f"**Std Dev:** {matches_df['MATCH_SCORE'].std():.2f}")
            st.write(f"**Min:** {matches_df['MATCH_SCORE'].min():.2f}")
            st.write(f"**Max:** {matches_df['MATCH_SCORE'].max():.2f}")

        with col2:
            st.markdown("#### Match Type Analysis")
            match_type_stats = matches_df['MATCH_TYPE'].value_counts()
            for match_type, count in match_type_stats.items():
                percentage = count / len(matches_df) * 100
                st.write(f"**{match_type}:** {count} ({percentage:.1f}%)")

        with col3:
            st.markdown("#### Data Completeness")
            st.write(f"**Avg SPR Completeness:** {matches_df['COMPLETENESS_SPR'].mean():.1%}")
            st.write(f"**Avg CAD Completeness:** {matches_df['COMPLETENESS_CAD'].mean():.1%}")

            # Completeness correlation
            correlation = matches_df['COMPLETENESS_SPR'].corr(matches_df['COMPLETENESS_CAD'])
            st.write(f"**Completeness Correlation:** {correlation:.3f}")


def render_unmatched_analysis(matches_df, spr_processed):
    """Render unmatched records analysis"""
    st.subheader("🔍 Unmatched Records Analysis")

    matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique()) if len(matches_df) > 0 else set()
    unmatched_spr = spr_processed[~spr_processed.get('ADDRESS_ID', spr_processed.index).isin(matched_spr_ids)]

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Total Unmatched SPR Records:** {len(unmatched_spr)}")
        st.write(f"**Unmatched Rate:** {len(unmatched_spr) / len(spr_processed) * 100:.1f}%")

        if len(unmatched_spr) > 0:
            st.write("**Common Issues in Unmatched Records:**")
            low_completeness = len(unmatched_spr[unmatched_spr['COMPLETENESS_SCORE'] < 0.5])
            st.write(f"- Low completeness: {low_completeness} records")

            empty_streets = len(unmatched_spr[unmatched_spr['STREET_NORM'] == ''])
            st.write(f"- Empty street names: {empty_streets} records")

            empty_houses = len(unmatched_spr[unmatched_spr['HOUSE_NORM'] == ''])
            st.write(f"- Empty house numbers: {empty_houses} records")

    with col2:
        if len(unmatched_spr) > 0:
            st.write("**Sample Unmatched Records:**")
            sample_unmatched = unmatched_spr[['STREET_NAME', 'HOUSE', 'BUILDING', 'COMPLETENESS_SCORE']].head(10)
            st.dataframe(sample_unmatched, use_container_width=True)

    return unmatched_spr




def render_recommendations(matches_df, spr_processed, unmatched_spr):
    """Render recommendations"""
    st.subheader("💡 Recommendations")

    recommendations = []

    if len(matches_df) > 0:
        avg_score = matches_df['MATCH_SCORE'].mean()
        if avg_score < 85:
            recommendations.append("Consider lowering the matching threshold to capture more potential matches")

        low_quality_matches = len(matches_df[matches_df['MATCH_SCORE'] < 80])
        if low_quality_matches > len(matches_df) * 0.2:
            recommendations.append(
                "High number of low-quality matches - review and possibly adjust matching parameters")

    match_rate = len(matches_df) / len(spr_processed) if len(spr_processed) > 0 else 0
    if match_rate < 0.5:
        recommendations.append(
            "Low match rate - consider data quality improvements or relaxed matching criteria")

    if len(unmatched_spr) > 0:
        low_completeness_unmatched = len(unmatched_spr[unmatched_spr['COMPLETENESS_SCORE'] < 0.5])
        if low_completeness_unmatched > len(unmatched_spr) * 0.3:
            recommendations.append(
                "Many unmatched records have low completeness - focus on data quality improvement")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.success("No specific recommendations - matching performance looks good!")


def render_summary_report(matches_df, quality_metrics, spr_processed):
    """Render summary report"""
    st.subheader("📝 Summary Report")

    # Calculate values outside f-string to avoid formatting issues
    avg_score = matches_df['MATCH_SCORE'].mean() if len(matches_df) > 0 else 0
    avg_score_text = f"{avg_score:.1f}" if len(matches_df) > 0 else "N/A"
    high_quality_count = len(matches_df[matches_df['MATCH_SCORE'] >= 90]) if len(matches_df) > 0 else 0
    medium_quality_count = len(
        matches_df[(matches_df['MATCH_SCORE'] >= 80) & (matches_df['MATCH_SCORE'] < 90)]) if len(
        matches_df) > 0 else 0
    low_quality_count = len(matches_df[matches_df['MATCH_SCORE'] < 80]) if len(matches_df) > 0 else 0

    match_rate = len(matches_df) / len(spr_processed) if len(spr_processed) > 0 else 0
    
    # Generate recommendations
    recommendations = []
    if len(matches_df) > 0:
        if avg_score < 85:
            recommendations.append("Consider lowering the matching threshold to capture more potential matches")

    summary_text = f"""
    # Address Matching Summary Report

    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    **Method:** {quality_metrics.get('matching_method', 'Unknown')}
    **Processing Time:** {quality_metrics.get('processing_time', 0):.2f} seconds

    ## Results Overview
    - **Total SPR Records:** {len(spr_processed):,}
    - **Total Matches:** {len(matches_df):,}
    - **Match Rate:** {match_rate:.1%}
    - **Average Score:** {avg_score_text}

    ## Quality Assessment
    - **High Quality Matches (≥90):** {high_quality_count}
    - **Medium Quality Matches (80-89):** {medium_quality_count}
    - **Low Quality Matches (<80):** {low_quality_count}

    ## Recommendations
    {chr(10).join(f"- {rec}" for rec in recommendations) if recommendations else "- No specific recommendations"}
    """

    st.markdown(summary_text)

    return summary_text


def render_unmatched_addresses_summary(unmatched_df):
    """Render unmatched addresses summary metrics"""
    if unmatched_df is None or unmatched_df.empty:
        st.info("No unmatched addresses found or data not available")
        return
    
    st.subheader("📊 Unmatched Addresses Summary")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Unmatched", f"{len(unmatched_df):,}")
    
    with col2:
        if 'IS_COMPLETE' in unmatched_df.columns:
            complete_count = unmatched_df['IS_COMPLETE'].sum()
            complete_pct = complete_count / len(unmatched_df) * 100 if len(unmatched_df) > 0 else 0
            st.metric("Complete Records", f"{complete_count:,} ({complete_pct:.1f}%)")
        else:
            st.metric("Complete Records", "N/A")
    
    with col3:
        if 'HAS_STREET' in unmatched_df.columns:
            missing_street = (~unmatched_df['HAS_STREET']).sum()
            st.metric("Missing Street", f"{missing_street:,}")
        else:
            st.metric("Missing Street", "N/A")
    
    with col4:
        if 'HAS_HOUSE' in unmatched_df.columns:
            missing_house = (~unmatched_df['HAS_HOUSE']).sum()
            st.metric("Missing House", f"{missing_house:,}")
        else:
            st.metric("Missing House", "N/A")


def render_unmatched_data_quality_analysis(unmatched_df):
    """Render data quality analysis for unmatched addresses"""
    if unmatched_df is None or unmatched_df.empty:
        return
    
    st.subheader("🔍 Data Quality Analysis")
    
    # Quality breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Completeness Breakdown")
        
        if 'COMPLETENESS_SCORE' in unmatched_df.columns:
            # Completeness categories
            excellent = unmatched_df[unmatched_df['COMPLETENESS_SCORE'] >= 0.9]
            good = unmatched_df[(unmatched_df['COMPLETENESS_SCORE'] >= 0.7) & (unmatched_df['COMPLETENESS_SCORE'] < 0.9)]
            fair = unmatched_df[(unmatched_df['COMPLETENESS_SCORE'] >= 0.5) & (unmatched_df['COMPLETENESS_SCORE'] < 0.7)]
            poor = unmatched_df[unmatched_df['COMPLETENESS_SCORE'] < 0.5]
            
            st.success(f"🟢 Excellent (≥90%): {len(excellent):,} records")
            st.info(f"🔵 Good (70-89%): {len(good):,} records")
            st.warning(f"🟡 Fair (50-69%): {len(fair):,} records")
            st.error(f"🔴 Poor (<50%): {len(poor):,} records")
        else:
            st.info("Completeness score not available")
    
    with col2:
        st.markdown("### Common Issues")
        
        # Field-specific analysis
        total = len(unmatched_df)
        
        if 'HAS_STREET' in unmatched_df.columns:
            missing_street = (~unmatched_df['HAS_STREET']).sum()
            st.write(f"**Missing Street Name:** {missing_street:,} ({missing_street/total*100:.1f}%)")
        
        if 'HAS_HOUSE' in unmatched_df.columns:
            missing_house = (~unmatched_df['HAS_HOUSE']).sum()
            st.write(f"**Missing House Number:** {missing_house:,} ({missing_house/total*100:.1f}%)")
        
        if 'HAS_BUILDING' in unmatched_df.columns:
            missing_building = (~unmatched_df['HAS_BUILDING']).sum()
            st.write(f"**Missing Building:** {missing_building:,} ({missing_building/total*100:.1f}%)")
        
        # Additional analysis
        if 'FULL_ADDRESS' in unmatched_df.columns:
            empty_normalized = unmatched_df[unmatched_df['FULL_ADDRESS'].str.strip() == '']
            st.write(f"**Empty Normalized Address:** {len(empty_normalized):,} ({len(empty_normalized)/total*100:.1f}%)")
        else:
            st.info("Address analysis not available")


def render_unmatched_addresses_filters(unmatched_df):
    """Render filters for unmatched addresses"""
    if unmatched_df is None or unmatched_df.empty:
        return unmatched_df
    
    st.subheader("🔧 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Completeness filter - only show if column exists
        if 'COMPLETENESS_SCORE' in unmatched_df.columns:
            completeness_options = st.multiselect(
                "Completeness Level",
                options=["Excellent (≥90%)", "Good (70-89%)", "Fair (50-69%)", "Poor (<50%)"],
                default=["Excellent (≥90%)", "Good (70-89%)", "Fair (50-69%)", "Poor (<50%)"]
            )
        else:
            completeness_options = []
            st.info("Completeness filter not available")
    
    with col2:
        # Missing data filter - only show relevant options
        filter_options = ["All Records"]
        if 'HAS_STREET' in unmatched_df.columns:
            filter_options.append("Missing Street")
        if 'HAS_HOUSE' in unmatched_df.columns:
            filter_options.append("Missing House")
        if 'HAS_BUILDING' in unmatched_df.columns:
            filter_options.append("Missing Building")
        if 'IS_COMPLETE' in unmatched_df.columns:
            filter_options.append("Complete Records Only")
        
        missing_data_filter = st.selectbox(
            "Missing Data Filter",
            options=filter_options
        )
    
    with col3:
        # Street name search
        street_search = st.text_input("Search Street Name", "")
    
    # Apply filters
    filtered_df = unmatched_df.copy()
    
    # Completeness filter
    if completeness_options and 'COMPLETENESS_SCORE' in filtered_df.columns:
        completeness_mask = pd.Series(False, index=filtered_df.index)
        if "Excellent (≥90%)" in completeness_options:
            completeness_mask |= filtered_df['COMPLETENESS_SCORE'] >= 0.9
        if "Good (70-89%)" in completeness_options:
            completeness_mask |= (filtered_df['COMPLETENESS_SCORE'] >= 0.7) & (filtered_df['COMPLETENESS_SCORE'] < 0.9)
        if "Fair (50-69%)" in completeness_options:
            completeness_mask |= (filtered_df['COMPLETENESS_SCORE'] >= 0.5) & (filtered_df['COMPLETENESS_SCORE'] < 0.7)
        if "Poor (<50%)" in completeness_options:
            completeness_mask |= filtered_df['COMPLETENESS_SCORE'] < 0.5
        
        filtered_df = filtered_df[completeness_mask]
    
    # Missing data filter
    if missing_data_filter == "Missing Street" and 'HAS_STREET' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['HAS_STREET']]
    elif missing_data_filter == "Missing House" and 'HAS_HOUSE' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['HAS_HOUSE']]
    elif missing_data_filter == "Missing Building" and 'HAS_BUILDING' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['HAS_BUILDING']]
    elif missing_data_filter == "Complete Records Only" and 'IS_COMPLETE' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['IS_COMPLETE']]
    
    # Street search filter
    if street_search and 'STREET_NAME' in filtered_df.columns:
        street_mask = filtered_df['STREET_NAME'].str.contains(street_search, case=False, na=False)
        filtered_df = filtered_df[street_mask]
    
    return filtered_df


def render_unmatched_addresses_table(filtered_df):
    """Render unmatched addresses table with pagination"""
    if filtered_df is None or filtered_df.empty:
        st.info("No unmatched addresses match the current filters")
        return
    
    st.subheader(f"📋 Unmatched Addresses ({len(filtered_df):,} records)")
    
    # Pagination
    page_size = 50
    total_pages = (len(filtered_df) - 1) // page_size + 1
    
    col1, col2 = st.columns([3, 1])
    with col1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
    with col2:
        st.metric("Total Pages", total_pages)
    
    # Calculate pagination
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    # Select columns for display based on what's available
    base_columns = ['STREET_NAME', 'HOUSE', 'BUILDING']
    analysis_columns = ['COMPLETENESS_SCORE', 'HAS_STREET', 'HAS_HOUSE', 'HAS_BUILDING']
    
    display_columns = []
    for col in base_columns:
        if col in page_df.columns:
            display_columns.append(col)
    
    # Add FULL_ADDRESS if available
    if 'FULL_ADDRESS' in page_df.columns:
        display_columns.append('FULL_ADDRESS')
    
    # Add analysis columns if available
    for col in analysis_columns:
        if col in page_df.columns:
            display_columns.append(col)
    
    if not display_columns:
        st.error("No displayable columns found")
        return
    
    # Create display dataframe
    display_df = page_df[display_columns].copy()
    
    # Format completeness score as percentage if available
    if 'COMPLETENESS_SCORE' in display_df.columns:
        display_df['COMPLETENESS_SCORE'] = display_df['COMPLETENESS_SCORE'].apply(lambda x: f"{x:.1%}")
    
    # Style the dataframe
    def highlight_completeness(val):
        if isinstance(val, str) and '%' in val:
            pct = float(val.replace('%', '')) / 100
            if pct >= 0.9:
                return 'background-color: #d4edda'
            elif pct >= 0.7:
                return 'background-color: #d1ecf1'
            elif pct >= 0.5:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        return ''
    
    def highlight_boolean(val):
        if val is True:
            return 'background-color: #d4edda'
        elif val is False:
            return 'background-color: #f8d7da'
        return ''
    
    # Apply styling
    styled_df = display_df.style
    
    # Apply completeness highlighting if column exists
    if 'COMPLETENESS_SCORE' in display_df.columns:
        styled_df = styled_df.applymap(highlight_completeness, subset=['COMPLETENESS_SCORE'])
    
    # Apply boolean highlighting if columns exist
    boolean_columns = [col for col in ['HAS_STREET', 'HAS_HOUSE', 'HAS_BUILDING'] if col in display_df.columns]
    if boolean_columns:
        styled_df = styled_df.applymap(highlight_boolean, subset=boolean_columns)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Show pagination info
    st.info(f"Showing records {start_idx + 1:,} to {min(end_idx, len(filtered_df)):,} of {len(filtered_df):,}")



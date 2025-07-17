import pandas as pd
from sqlalchemy import create_engine, text
from rapidfuzz import process, fuzz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import re
import time
import logging
from datetime import datetime
import json
import base64
from io import BytesIO
import zipfile
import os

# Import decomposed modules
from src.database.connection import get_database_connection
from src.database.data_loader import load_registry_data
from src.matching.advanced_matcher import AdvancedAddressMatcher
from src.utils.preprocessing import preprocess_registries
from src.quality.analyzer import analyze_data_quality
from src.visualization.charts import create_match_quality_chart, create_data_quality_dashboard
from src.utils.export import create_export_package

# ---------------- Configuration ----------------
st.set_page_config(
    page_title="Address Registry Matcher",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏘️ Advanced Address Registry Matcher</h1>
        <p>Comprehensive address matching and mapping between SPR and Cadastre registries</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "matching_results" not in st.session_state:
        st.session_state.matching_results = None
    if "quality_metrics" not in st.session_state:
        st.session_state.quality_metrics = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Configuration")

        # Database settings
        st.subheader("Database Settings")

        # Allow environment variables for production
        default_host = os.getenv('DB_HOST', 'localhost')
        default_port = int(os.getenv('DB_PORT', '3306'))
        default_db = os.getenv('DB_NAME', 'experiments')
        default_user = os.getenv('DB_USER', 'root')
        default_password = os.getenv('DB_PASSWORD', 'DC123456')

        db_host = st.text_input("Database Host", value=default_host)
        db_port = st.number_input("Database Port", value=default_port)
        db_name = st.text_input("Database Name", value=default_db)
        db_user = st.text_input("Database User", value=default_user)
        db_password = st.text_input("Database Password", value=default_password, type="password")

        # Table settings
        st.subheader("Table Configuration")
        spr_table = st.text_input("SPR Table Name", value="spr")
        cad_table = st.text_input("Cadastre Table Name", value="cadastre_dp")

        # Matching parameters
        st.subheader("Matching Parameters")
        matching_method = st.selectbox(
            "Matching Strategy",
            ["Exact Only", "Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]
        )

        # Show performance warning for fuzzy matching
        if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
            st.warning("⚠️ **Performance Note:** Fuzzy matching can be slow, especially with large datasets. "
                       "Consider starting with a smaller record count for testing.")

        # Processing limits
        st.subheader("Processing Configuration")
        col1, col2 = st.columns(2)

        with col1:
            use_all_records = st.checkbox(
                "Process All Records",
                value=False,
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
                max_value=5000,
                value=500,
                step=50,
                help="Number of records to process in each chunk (affects memory usage and progress updates)"
            )

            # Show chunk size recommendations
            if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                if chunk_size > 1000:
                    st.info(
                        "💡 **Tip:** Smaller chunks (100-500) provide better progress tracking for fuzzy matching")

        # Fuzzy matching parameters
        if matching_method != "Exact Only":
            st.subheader("Fuzzy Matching Settings")
            col1, col2 = st.columns(2)

            with col1:
                threshold = st.slider(
                    "Fuzzy Match Threshold",
                    50, 100, 85,
                    help="Minimum similarity score for fuzzy matches (higher = stricter)"
                )

            with col2:
                # Performance estimate
                estimated_records = max_records if not use_all_records else len(
                    spr_processed) if 'spr_processed' in locals() else 0
                if estimated_records > 50000:
                    st.error(
                        "🚨 **High Volume Warning:** Processing >50k records with fuzzy matching may take significant time")
                elif estimated_records > 20000:
                    st.warning("⚠️ **Medium Volume:** Processing >20k records may take several minutes")
                else:
                    st.info("✅ **Good Volume:** Processing should complete reasonably quickly")

        # Advanced options
        st.subheader("Advanced Options")
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)
        export_unmatched = st.checkbox("Include Unmatched Records", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Overview", "🔍 Matching Process", "📈 Results Analysis", "📋 Quality Report"])

    with tab1:
        st.subheader("Registry Data Overview")

        # Load database connection
        engine = get_database_connection(db_host, db_port, db_name, db_user, db_password)
        if engine is None:
            st.error("Cannot proceed without database connection")
            return

        # Load data
        with st.spinner("Loading registry data..."):
            spr_df = load_registry_data("SPR", spr_table, engine)
            cad_df = load_registry_data("Cadastre", cad_table, engine)

        if spr_df is None or cad_df is None:
            st.error("Failed to load registry data")
            return

        # Preprocess data
        with st.spinner("Preprocessing data..."):
            spr_processed, cad_processed = preprocess_registries(spr_df, cad_df)

        # Data quality analysis
        spr_quality = analyze_data_quality(spr_processed, "SPR")
        cad_quality = analyze_data_quality(cad_processed, "Cadastre")

        # Display metrics with processing limits info
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

        # Processing limits info
        if 'max_records' in locals() and 'use_all_records' in locals():
            if not use_all_records and max_records:
                processing_limit = min(max_records, len(spr_processed))
                st.info(
                    f"🔢 **Processing Configuration:** {processing_limit:,} records will be processed in chunks of {chunk_size:,}")

                # Performance estimate for street fuzzy matching (much faster)
                if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                    estimated_time = processing_limit * 0.002  # Much faster: 2ms per record
                    st.info(
                        f"⏱️ **Estimated Time:** ~{estimated_time / 60:.1f} minutes for street-only fuzzy matching")
            else:
                st.info(
                    f"🔢 **Processing Configuration:** All {len(spr_processed):,} records will be processed in chunks of {chunk_size:,}")

                # Performance warning for large datasets (updated for street-only matching)
                if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                    if len(spr_processed) > 100000:
                        estimated_time = len(spr_processed) * 0.002
                        st.warning(
                            f"⚠️ **Performance Note:** Processing {len(spr_processed):,} records may take ~{estimated_time / 60:.1f} minutes")
                    else:
                        st.success("✅ **Performance:** Street-only fuzzy matching should be quite efficient")

        # Chunk size info
        if 'chunk_size' in locals():
            if chunk_size < 100:
                st.info("💡 **Small chunks:** More frequent progress updates, slightly slower overall")
            elif chunk_size > 1000:
                st.info("💡 **Large chunks:** Faster processing, less frequent progress updates")

        # Data quality visualization
        quality_fig = create_data_quality_dashboard(spr_quality, cad_quality)
        st.plotly_chart(quality_fig, use_container_width=True)

        # Sample data preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SPR Sample Data")
            st.dataframe(
                spr_processed[['STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))

        with col2:
            st.subheader("Cadastre Sample Data")
            st.dataframe(
                cad_processed[['STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))

    with tab2:
        st.subheader("Address Matching Process")

        if 'spr_processed' not in locals() or 'cad_processed' not in locals():
            st.warning("Please load data in the Data Overview tab first")
            return

        # Matching controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button("🚀 Start Matching Process", type="primary", use_container_width=True):
                st.session_state.stop_requested = False

                try:
                    # Determine processing configuration
                    if use_all_records:
                        processing_records = len(spr_processed)
                        matcher = AdvancedAddressMatcher(spr_processed, cad_processed)
                        st.info(
                            f"🔢 **Processing:** All {processing_records:,} SPR records in chunks of {chunk_size:,}")
                    else:
                        processing_records = min(max_records, len(spr_processed))
                        matcher = AdvancedAddressMatcher(spr_processed, cad_processed, max_records)
                        st.info(
                            f"🔢 **Processing:** {processing_records:,} of {len(spr_processed):,} SPR records in chunks of {chunk_size:,}")

                    # Performance warning for fuzzy matching
                    if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        if processing_records > 50000:
                            st.error(
                                "🚨 **High Volume Alert:** This may take 30+ minutes. Consider reducing record count for initial testing.")
                        elif processing_records > 20000:
                            st.warning(
                                "⚠️ **Medium Volume Alert:** This may take 10-20 minutes. Please be patient.")
                        elif processing_records > 5000:
                            st.info("ℹ️ **Processing Alert:** This may take 2-5 minutes for fuzzy matching.")

                    # Initialize progress tracking
                    overall_start_time = time.time()

                    # Create overall progress container
                    overall_progress_container = st.container()
                    with overall_progress_container:
                        st.subheader("📊 Overall Matching Progress")

                        overall_progress = st.progress(0)
                        overall_status = st.empty()

                        # Phase indicators
                        phase_cols = st.columns(3)
                        with phase_cols[0]:
                            exact_phase = st.empty()
                        with phase_cols[1]:
                            fuzzy_phase = st.empty()
                        with phase_cols[2]:
                            complete_phase = st.empty()

                    all_matches = []
                    phase_progress = 0
                    total_phases = 0

                    # Determine total phases
                    if matching_method in ["Exact Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        total_phases += 1
                    if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        total_phases += 1

                    # Phase 1: Exact matching
                    if matching_method in ["Exact Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        exact_phase.markdown("🔄 **Phase 1: Exact Matching** - In Progress")
                        overall_status.text(f"Phase 1/{total_phases}: Finding exact matches...")

                        exact_matches = matcher.find_exact_matches()
                        all_matches.append(exact_matches)

                        phase_progress += 1
                        overall_progress.progress(phase_progress / total_phases)
                        exact_phase.markdown("✅ **Phase 1: Exact Matching** - Complete")

                        st.success(f"✅ Phase 1 Complete: Found {len(exact_matches)} exact matches")

                    # Phase 2: Fuzzy matching
                    if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        exclude_ids = set()
                        if len(all_matches) > 0:
                            exclude_ids = set(all_matches[0]['ADDRESS_ID_SPR'].unique())

                        fuzzy_phase.markdown("🔄 **Phase 2: Fuzzy Matching** - In Progress")
                        overall_status.text(
                            f"Phase 2/{total_phases}: Running fuzzy matching (this may take time)...")

                        # Additional fuzzy matching warning
                        remaining_records = processing_records - len(exclude_ids)
                        if remaining_records > 10000:
                            st.warning(
                                f"⚠️ **Fuzzy Matching:** Processing {remaining_records:,} remaining records. This phase may take several minutes.")

                        fuzzy_matches = matcher.find_fuzzy_matches(
                            threshold=threshold,
                            chunk_size=chunk_size,
                            exclude_spr_ids=exclude_ids
                        )
                        all_matches.append(fuzzy_matches)

                        phase_progress += 1
                        overall_progress.progress(phase_progress / total_phases)
                        fuzzy_phase.markdown("✅ **Phase 2: Fuzzy Matching** - Complete")

                        st.success(f"✅ Phase 2 Complete: Found {len(fuzzy_matches)} fuzzy matches")

                    # Combine results
                    if all_matches:
                        final_matches = pd.concat([df for df in all_matches if not df.empty], ignore_index=True)
                    else:
                        final_matches = pd.DataFrame()

                    # Final processing
                    complete_phase.markdown("🔄 **Phase 3: Finalizing** - In Progress")
                    overall_status.text("Finalizing results and generating reports...")

                    end_time = time.time()
                    processing_time = end_time - overall_start_time

                    # Store results with processing info
                    st.session_state.matching_results = final_matches
                    st.session_state.processing_time = processing_time
                    st.session_state.quality_metrics = {
                        'spr_quality': spr_quality,
                        'cad_quality': cad_quality,
                        'processing_time': processing_time,
                        'matching_method': matching_method,
                        'processing_records': processing_records,
                        'total_spr_records': len(spr_processed),
                        'records_processed_percentage': processing_records / len(spr_processed) * 100,
                        'chunk_size': chunk_size
                    }

                    # Final updates
                    overall_progress.progress(1.0)
                    complete_phase.markdown("✅ **Phase 3: Finalizing** - Complete")
                    overall_status.text("🎉 All phases completed successfully!")

                    # Display comprehensive summary
                    st.balloons()

                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    with summary_col1:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with summary_col2:
                        st.metric("Records Processed", f"{processing_records:,}")
                    with summary_col3:
                        st.metric("Total Matches", f"{len(final_matches):,}")
                    with summary_col4:
                        match_rate = len(final_matches) / processing_records * 100 if processing_records > 0 else 0
                        st.metric("Match Rate", f"{match_rate:.1f}%")

                    # Performance summary
                    processing_speed = processing_records / processing_time if processing_time > 0 else 0
                    coverage_info = f"({processing_records:,} of {len(spr_processed):,} total records)" if processing_records < len(
                        spr_processed) else "(all records)"

                    st.info(f"""
                    🎯 **Matching Summary:**
                    - **Records Processed:** {processing_records:,} {coverage_info}
                    - **Chunk Size:** {chunk_size:,} records per chunk
                    - **Matches Found:** {len(final_matches):,}
                    - **Match Rate:** {match_rate:.1f}%
                    - **Processing Speed:** {processing_speed:.1f} records/second
                    - **Method Used:** {matching_method}
                    - **Coverage:** {processing_records / len(spr_processed) * 100:.1f}% of total SPR records
                    """)

                    # Show limitation warning if applicable
                    if processing_records < len(spr_processed):
                        st.warning(
                            f"⚠️ **Note:** Only {processing_records:,} out of {len(spr_processed):,} total SPR records were processed. "
                            f"To process all records, check 'Process All Records' option.")

                except Exception as e:
                    st.error(f"Error during matching process: {str(e)}")
                    logger.error(f"Matching process error: {str(e)}")
                    st.exception(e)  # Show full traceback in development

        with col2:
            if st.button("⏹ Stop Process", use_container_width=True):
                st.session_state.stop_requested = True
                st.warning("Stop requested...")

        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.matching_results = None
                st.session_state.quality_metrics = None
                st.session_state.stop_requested = False
                st.rerun()

        # Display current matching progress/results
        if st.session_state.matching_results is not None:
            matches_df = st.session_state.matching_results
            quality_metrics = st.session_state.quality_metrics

            # Get processing info
            processing_records = quality_metrics.get('processing_records', len(spr_processed))
            total_records = quality_metrics.get('total_spr_records', len(spr_processed))
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

            # Match type breakdown
            if len(matches_df) > 0:
                match_type_counts = matches_df['MATCH_TYPE'].value_counts()
                st.subheader("Match Type Distribution")

                # Create columns for each match type
                cols = st.columns(len(match_type_counts))
                for i, (match_type, count) in enumerate(match_type_counts.items()):
                    with cols[i]:
                        percentage = count / len(matches_df) * 100
                        st.metric(match_type, f"{count} ({percentage:.1f}%)")

    with tab3:
        st.subheader("Results Analysis & Visualization")

        if st.session_state.matching_results is None:
            st.info("No matching results available. Please run the matching process first.")
            return

        matches_df = st.session_state.matching_results

        if len(matches_df) == 0:
            st.warning("No matches found. Try adjusting the matching parameters.")
            return

        # Create comprehensive visualization
        quality_chart = create_match_quality_chart(matches_df)
        if quality_chart:
            st.plotly_chart(quality_chart, use_container_width=True)

        # Detailed match analysis
        st.subheader("Detailed Match Analysis")

        # Score distribution analysis
        col1, col2 = st.columns(2)

        with col1:
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

        with col2:
            st.subheader("Top Scoring Matches")
            top_matches = matches_df.nlargest(10, 'MATCH_SCORE')[
                ['STREET_NAME_SPR', 'HOUSE_SPR', 'STREET_NAME_CAD', 'HOUSE_CAD', 'MATCH_SCORE']
            ]
            st.dataframe(top_matches, use_container_width=True)

        # Interactive match explorer
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

        # Manual review interface
        st.subheader("Manual Review Interface")

        if len(filtered_matches) > 0:
            review_idx = st.selectbox(
                "Select match to review",
                range(len(filtered_matches)),
                format_func=lambda
                    x: f"Match {x + 1}: {filtered_matches.iloc[x]['STREET_NAME_SPR']} → {filtered_matches.iloc[x]['STREET_NAME_CAD']} (Score: {filtered_matches.iloc[x]['MATCH_SCORE']})"
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

    with tab4:
        st.subheader("Quality Report & Export")

        if st.session_state.matching_results is None:
            st.info("No matching results available. Please run the matching process first.")
            return

        matches_df = st.session_state.matching_results
        quality_metrics = st.session_state.quality_metrics

        # Comprehensive quality report
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
                st.write(
                    f"**High Quality Matches (≥90):** {high_quality} ({high_quality / len(matches_df) * 100:.1f}%)")
                st.write(
                    f"**Low Quality Matches (<80):** {low_quality} ({low_quality / len(matches_df) * 100:.1f}%)")

                # Quality assessment
                if avg_score >= 90:
                    st.success("🟢 Excellent matching quality")
                elif avg_score >= 80:
                    st.info("🔵 Good matching quality")
                elif avg_score >= 70:
                    st.warning("🟡 Fair matching quality")
                else:
                    st.error("🔴 Poor matching quality - review parameters")

        # Detailed statistics
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

        # Unmatched records analysis
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
                sample_unmatched = unmatched_spr[['STREET_NAME', 'HOUSE', 'BUILDING', 'COMPLETENESS_SCORE']].head(
                    10)
                st.dataframe(sample_unmatched, use_container_width=True)

        # Export section
        st.subheader("📥 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export Matched Records"):
                csv_data = matches_df.to_csv(index=False)
                st.download_button(
                    label="Download Matched Records CSV",
                    data=csv_data,
                    file_name=f"matched_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📋 Export Unmatched Records") and len(unmatched_spr) > 0:
                unmatched_csv = unmatched_spr.to_csv(index=False)
                st.download_button(
                    label="Download Unmatched Records CSV",
                    data=unmatched_csv,
                    file_name=f"unmatched_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col3:
            if st.button("📦 Export Complete Package"):
                # Create comprehensive export package
                export_data = create_export_package(matches_df, spr_processed, cad_processed, quality_metrics)
                st.download_button(
                    label="Download Complete Package",
                    data=export_data,
                    file_name=f"address_matching_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

        # Recommendations
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

        # Generate summary report
        st.subheader("📝 Summary Report")

        # Calculate values outside f-string to avoid formatting issues
        avg_score = matches_df['MATCH_SCORE'].mean() if len(matches_df) > 0 else 0
        avg_score_text = f"{avg_score:.1f}" if len(matches_df) > 0 else "N/A"
        high_quality_count = len(matches_df[matches_df['MATCH_SCORE'] >= 90]) if len(matches_df) > 0 else 0
        medium_quality_count = len(
            matches_df[(matches_df['MATCH_SCORE'] >= 80) & (matches_df['MATCH_SCORE'] < 90)]) if len(
            matches_df) > 0 else 0
        low_quality_count = len(matches_df[matches_df['MATCH_SCORE'] < 80]) if len(matches_df) > 0 else 0

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

        # Export summary report
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"matching_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


# Run the application
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import logging
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MVC components
from models import analyze_data_quality, preprocess_registries, load_registry_data_from_csv
from views import (
    configure_page, apply_custom_css, render_main_header, render_sidebar_config,
    render_data_overview_metrics, render_sample_data_preview,
    render_score_quality_categories, render_top_scoring_matches,
    render_interactive_match_explorer, render_manual_review_interface,
    render_quality_report_summary, render_detailed_statistics,
    render_unmatched_analysis, render_export_options,
    render_recommendations, render_summary_report,
    create_data_quality_dashboard, create_match_quality_chart
)
from controllers.matching_controller import MatchingController
from controllers.duplicates_controller import DuplicatesController
from config import setup_logging, get_app_config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    
    # Configure page
    configure_page()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render main header
    render_main_header()
    
    # Initialize controllers
    controller = MatchingController()
    duplicates_controller = DuplicatesController()
    
    # Render sidebar configuration
    config = render_sidebar_config()
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Data Overview", "🔍 Matching Process", "📈 Results Analysis", "📋 Quality Report", "🔍 SPR Duplicates"]
    )
    
    with tab1:
        render_data_overview_tab(controller, config)
    
    with tab2:
        render_matching_process_tab(controller, config)
    
    with tab3:
        render_results_analysis_tab(controller)
    
    with tab4:
        render_quality_report_tab(controller)
    
    with tab5:
        render_duplicates_tab(duplicates_controller)


def render_data_overview_tab(controller: MatchingController, config: Dict[str, Any]):
    """Render the data overview tab"""
    st.subheader("Registry Data Overview")
    
    # Load data
    spr_processed, cad_processed = controller.load_data(config)
    
    if spr_processed is None or cad_processed is None:
        st.error("Failed to load registry data")
        return
    
    # Get data quality metrics
    quality_metrics = controller.get_data_quality_metrics(spr_processed, cad_processed)
    spr_quality = quality_metrics['spr_quality']
    cad_quality = quality_metrics['cad_quality']
    
    # Render data overview metrics
    render_data_overview_metrics(spr_quality, cad_quality)
    
    # Show processing configuration info
    show_processing_info(config, spr_processed)
    
    # Data quality visualization
    quality_fig = create_data_quality_dashboard(spr_quality, cad_quality)
    st.plotly_chart(quality_fig, use_container_width=True)
    
    # Sample data preview
    render_sample_data_preview(spr_processed, cad_processed)


def show_processing_info(config: Dict[str, Any], spr_processed: pd.DataFrame):
    """Show processing configuration information"""
    use_all_records = config.get('use_all_records', True)
    max_records = config.get('max_records', None)
    chunk_size = config.get('chunk_size', 500)
    
    if not use_all_records and max_records:
        processing_limit = min(max_records, len(spr_processed))
        st.info(f"🔢 **Processing Configuration:** {processing_limit:,} records will be processed using exact matching")
    else:
        st.info(f"🔢 **Processing Configuration:** All {len(spr_processed):,} records will be processed using exact matching")
        
    st.success("✅ **Performance:** Exact matching is fast and efficient for all dataset sizes")


def render_matching_process_tab(controller: MatchingController, config: Dict[str, Any]):
    """Render the matching process tab"""
    controller.render_matching_process_tab(config)


def render_results_analysis_tab(controller: MatchingController):
    """Render the results analysis tab"""
    st.subheader("Results Analysis & Visualization")
    
    matches_df = controller.get_matching_results()
    
    if matches_df is None:
        st.info("No matching results available. Please run the matching process first.")
        return
    
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
        render_score_quality_categories(matches_df)
    
    with col2:
        render_top_scoring_matches(matches_df)
    
    # Interactive match explorer
    filtered_matches = render_interactive_match_explorer(matches_df)
    
    # Manual review interface
    render_manual_review_interface(filtered_matches)


def render_quality_report_tab(controller: MatchingController):
    """Render the quality report tab"""
    st.subheader("Quality Report & Export")
    
    matches_df = controller.get_matching_results()
    quality_metrics = controller.get_quality_metrics()
    
    if matches_df is None:
        st.info("No matching results available. Please run the matching process first.")
        return
    
    spr_processed = st.session_state.get('spr_processed')
    if spr_processed is None:
        st.error("SPR data not available")
        return
    
    # Render quality report summary
    render_quality_report_summary(matches_df, quality_metrics, spr_processed)
    
    # Render detailed statistics
    render_detailed_statistics(matches_df)
    
    # Render unmatched analysis
    unmatched_spr = render_unmatched_analysis(matches_df, spr_processed)
    
    # Render export options
    render_export_options(matches_df, unmatched_spr)
    
    # Render recommendations
    render_recommendations(matches_df, spr_processed, unmatched_spr)
    
    # Render summary report
    render_summary_report(matches_df, quality_metrics, spr_processed)


def render_duplicates_tab(duplicates_controller: DuplicatesController):
    """Render the duplicates analysis tab"""
    duplicates_controller.render_duplicates_tab()


if __name__ == "__main__":
    main()
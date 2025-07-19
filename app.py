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
    configure_page, apply_custom_css, render_main_header,
    render_data_overview_metrics, render_sample_data_preview,
    render_interactive_match_explorer, render_manual_review_interface,
    render_combined_unmatched_summary, render_registry_selector, render_unmatched_addresses_table_combined
)
from controllers.matching_controller import MatchingController
from controllers.duplicates_controller import DuplicatesController




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
    
    
    # Create main tabs
    tab1, tab2, tab3, tab5, tab6 = st.tabs(
        ["📊 Data Overview", "🔍 Matching Process", "📈 Results Analysis",  "🔍 SPR Duplicates", "❌ Unmatched Addresses"]
    )
    
    with tab1:
        render_data_overview_tab(controller)
    
    with tab2:
        render_matching_process_tab(controller)
    
    with tab3:
        render_results_analysis_tab(controller)
    
    with tab5:
        render_duplicates_tab(duplicates_controller)
    
    with tab6:
        render_unmatched_addresses_tab(controller)


def render_data_overview_tab(controller: MatchingController):
    """Render the data overview tab"""
    st.subheader("Registry Data Overview")
    
    # Load data
    spr_processed, cad_processed = controller.load_data()
    
    if spr_processed is None or cad_processed is None:
        st.error("Failed to load registry data")
        return

    # Get data quality metrics
    quality_metrics = controller.get_data_quality_metrics(spr_processed, cad_processed)
    spr_quality = quality_metrics['spr_quality']
    cad_quality = quality_metrics['cad_quality']

    # Render data overview metrics
    render_data_overview_metrics(spr_quality, cad_quality)
    

    
    # Sample data preview
    render_sample_data_preview(spr_processed, cad_processed)




def render_matching_process_tab(controller: MatchingController):
    """Render the matching process tab"""
    controller.render_matching_process_tab()


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




    # Interactive match explorer
    filtered_matches = render_interactive_match_explorer(matches_df)
    
    # Manual review interface
    render_manual_review_interface(filtered_matches)



def render_duplicates_tab(duplicates_controller: DuplicatesController):
    """Render the duplicates analysis tab"""
    duplicates_controller.render_duplicates_tab()


def render_unmatched_addresses_tab(controller: MatchingController):
    """Render the unmatched addresses tab for both SPR and Cadastre"""
    st.subheader("Unmatched Addresses - SPR & Cadastre")
    
    # Check if data is loaded
    if st.session_state.get('spr_processed') is None or st.session_state.get('cad_processed') is None:
        st.warning("⚠️ Please load data in the **Data Overview** tab first")
        return
    
    # Check if matching has been performed
    if st.session_state.get('matching_results') is None:
        st.info("🔍 Please run the matching process in the **Matching Process** tab first to see unmatched addresses")
        
        # Show some basic info about total records
        spr_processed = st.session_state.get('spr_processed')
        cad_processed = st.session_state.get('cad_processed')
        if spr_processed is not None and cad_processed is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📋 **Total SPR records loaded:** {len(spr_processed):,}")
            with col2:
                st.write(f"🏛️ **Total Cadastre records loaded:** {len(cad_processed):,}")
            st.write("Once matching is complete, unmatched addresses from both registries will be displayed here for analysis.")
        return
    
    # Get unmatched addresses from both registries
    unmatched_spr_df = controller.get_unmatched_spr_addresses()
    unmatched_cad_df = controller.get_unmatched_cad_addresses()
    
    if unmatched_spr_df is None and unmatched_cad_df is None:
        st.error("❌ Unable to retrieve unmatched addresses")
        return
    
    # Check if both are empty
    spr_empty = unmatched_spr_df is None or unmatched_spr_df.empty
    cad_empty = unmatched_cad_df is None or unmatched_cad_df.empty
    
    if spr_empty and cad_empty:
        st.success("🎉 Excellent! No unmatched addresses found! All addresses from both registries have been successfully matched.")
        
        # Show some stats
        matches_df = st.session_state.get('matching_results')
        spr_processed = st.session_state.get('spr_processed')
        cad_processed = st.session_state.get('cad_processed')
        if matches_df is not None and spr_processed is not None and cad_processed is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📋 **Total SPR Records:** {len(spr_processed):,}")
            with col2:
                st.write(f"🏛️ **Total Cadastre Records:** {len(cad_processed):,}")
            st.write(f"✅ **Total Matches:** {len(matches_df):,}")
            st.write(f"🎯 **Match Rate:** 100%")
        return
    
    # Show summary with match context
    matches_df = st.session_state.get('matching_results')
    spr_processed = st.session_state.get('spr_processed')
    cad_processed = st.session_state.get('cad_processed')
    if matches_df is not None and spr_processed is not None and cad_processed is not None:
        match_rate = len(matches_df) / len(spr_processed) * 100 if len(spr_processed) > 0 else 0
        spr_unmatched_count = len(unmatched_spr_df) if unmatched_spr_df is not None else 0
        cad_unmatched_count = len(unmatched_cad_df) if unmatched_cad_df is not None else 0
        st.info(f"📊 **Matching Summary:** {len(matches_df):,} matches found ({match_rate:.1f}% match rate) | SPR: {spr_unmatched_count:,} unmatched | Cadastre: {cad_unmatched_count:,} unmatched")
    
    # Render combined summary
    render_combined_unmatched_summary(unmatched_spr_df, unmatched_cad_df)
    
    # Render registry selector
    registry_filter = render_registry_selector()
    
    # Render combined table with filtering
    render_unmatched_addresses_table_combined(unmatched_spr_df, unmatched_cad_df, registry_filter)
    


if __name__ == "__main__":
    main()
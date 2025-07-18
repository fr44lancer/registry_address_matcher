import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from models.address_models import (
    AdvancedAddressMatcher,
    load_registry_data_from_csv,
    preprocess_registries,
    analyze_data_quality
)
from views.ui_components import (
    render_matching_controls,
    render_matching_results_summary,
    render_match_type_distribution
)

logger = logging.getLogger(__name__)


class MatchingController:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "stop_requested" not in st.session_state:
            st.session_state.stop_requested = False
        if "matching_results" not in st.session_state:
            st.session_state.matching_results = None
        if "quality_metrics" not in st.session_state:
            st.session_state.quality_metrics = None
        if "spr_processed" not in st.session_state:
            st.session_state.spr_processed = None
        if "cad_processed" not in st.session_state:
            st.session_state.cad_processed = None

    def load_data(self, config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load and preprocess registry data"""
        try:
            with st.spinner("Loading registry data..."):
                # Load from CSV files
                spr_csv = "data/spr.csv"
                cad_csv = "data/cadastre.csv"

                spr_df = load_registry_data_from_csv("SPR", spr_csv)
                cad_df = load_registry_data_from_csv("Cadastre", cad_csv)

                if spr_df.empty or cad_df.empty:
                    st.error("Failed to load registry data")
                    return None, None

                # Preprocess data
                with st.spinner("Preprocessing data..."):
                    spr_processed, cad_processed = preprocess_registries(spr_df, cad_df)

                # Store in session state
                st.session_state.spr_processed = spr_processed
                st.session_state.cad_processed = cad_processed

                return spr_processed, cad_processed

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Data loading error: {str(e)}")
            return None, None

    def get_data_quality_metrics(self, spr_processed: pd.DataFrame, cad_processed: pd.DataFrame) -> Dict[str, Any]:
        """Get data quality metrics for both registries"""
        spr_quality = analyze_data_quality(spr_processed, "SPR")
        cad_quality = analyze_data_quality(cad_processed, "Cadastre")
        
        return {
            'spr_quality': spr_quality,
            'cad_quality': cad_quality
        }

    def handle_matching_process(self, config: Dict[str, Any], spr_processed: pd.DataFrame, cad_processed: pd.DataFrame):
        """Handle the complete matching process - exact matching only"""
        try:
            # Reset stop flag
            st.session_state.stop_requested = False

            # Determine processing configuration
            use_all_records = config['use_all_records']
            max_records = config['max_records']

            if use_all_records:
                processing_records = len(spr_processed)
                matcher = AdvancedAddressMatcher(spr_processed, cad_processed)
                st.info(f"🔢 **Processing:** All {processing_records:,} SPR records using exact matching")
            else:
                processing_records = min(max_records, len(spr_processed))
                matcher = AdvancedAddressMatcher(spr_processed, cad_processed, max_records)
                st.info(f"🔢 **Processing:** {processing_records:,} of {len(spr_processed):,} SPR records using exact matching")

            # Initialize progress tracking
            overall_start_time = time.time()
            
            # Simple progress display
            st.subheader("📊 Exact Matching")
            
            exact_matches = matcher.find_exact_matches()
            
            # Calculate processing time
            processing_time = time.time() - overall_start_time
            
            # Store results
            st.session_state.matching_results = exact_matches
            st.session_state.processing_time = processing_time
            st.session_state.quality_metrics = {
                'spr_quality': analyze_data_quality(st.session_state.spr_processed, "SPR"),
                'cad_quality': analyze_data_quality(st.session_state.cad_processed, "Cadastre"),
                'processing_time': processing_time,
                'matching_method': 'Exact Only',
                'processing_records': processing_records,
                'total_spr_records': len(spr_processed),
                'records_processed_percentage': processing_records / len(spr_processed) * 100
            }
            
            # Display results
            st.success(f"✅ Found {len(exact_matches)} exact matches in {processing_time:.2f}s")
            self._display_simple_summary(exact_matches, processing_time, processing_records, len(spr_processed))
            
            final_matches = exact_matches

            return final_matches

        except Exception as e:
            st.error(f"Error during matching process: {str(e)}")
            st.exception(e)
            return pd.DataFrame()


    def _display_simple_summary(self, matches, processing_time, processing_records, total_spr_records):
        """Display simple summary of matching results"""
        if matches.empty:
            st.info("No exact matches found")
            return
            
        match_rate = len(matches) / processing_records * 100 if processing_records > 0 else 0
        processing_speed = processing_records / processing_time if processing_time > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("Records Processed", f"{processing_records:,}")
        with col3:
            st.metric("Exact Matches", f"{len(matches):,}")
        with col4:
            st.metric("Match Rate", f"{match_rate:.1f}%")
            
        if processing_records < total_spr_records:
            st.info(f"📊 Processed {processing_records:,} of {total_spr_records:,} total records")


    def handle_stop_request(self):
        """Handle stop request from user"""
        st.session_state.stop_requested = True
        st.warning("Stop requested...")

    def handle_reset_request(self):
        """Handle reset request from user"""
        st.session_state.matching_results = None
        st.session_state.quality_metrics = None
        st.session_state.stop_requested = False
        st.session_state.spr_processed = None
        st.session_state.cad_processed = None
        st.rerun()

    def render_matching_process_tab(self, config: Dict[str, Any]):
        """Render the matching process tab"""
        st.subheader("Address Matching Process")

        # Check if data is loaded
        if st.session_state.spr_processed is None or st.session_state.cad_processed is None:
            st.warning("Please load data in the Data Overview tab first")
            return

        # Render matching controls
        start_button, stop_button, reset_button = render_matching_controls()

        # Handle button clicks
        if start_button:
            self.handle_matching_process(config, st.session_state.spr_processed, st.session_state.cad_processed)

        if stop_button:
            self.handle_stop_request()

        if reset_button:
            self.handle_reset_request()

        # Display current matching results if available
        if st.session_state.matching_results is not None:
            matches_df = st.session_state.matching_results
            quality_metrics = st.session_state.quality_metrics

            # Render results summary
            render_matching_results_summary(matches_df, quality_metrics)

            # Render match type distribution
            render_match_type_distribution(matches_df)

    def get_matching_results(self) -> Optional[pd.DataFrame]:
        """Get current matching results"""
        return st.session_state.matching_results

    def get_quality_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current quality metrics"""
        return st.session_state.quality_metrics

    def get_unmatched_spr_addresses(self) -> Optional[pd.DataFrame]:
        """Get unmatched SPR addresses - only available after matching process"""
        if st.session_state.spr_processed is None:
            return None
            
        # Only show unmatched addresses if matching has been performed
        matches_df = st.session_state.matching_results
        if matches_df is None:
            # No matching has been performed yet
            return None
            
        spr_processed = st.session_state.spr_processed
        
        if matches_df.empty:
            # If no matches found, all SPR addresses are unmatched
            unmatched_spr = spr_processed.copy()
        else:
            # Debug: Let's see what we're working with
            print(f"DEBUG: matches_df has {len(matches_df)} rows")
            print(f"DEBUG: spr_processed has {len(spr_processed)} rows") 
            print(f"DEBUG: matches_df columns: {matches_df.columns.tolist()}")
            print(f"DEBUG: spr_processed columns: {spr_processed.columns.tolist()}")
            
            # Get matched SPR IDs - handle both cases: with and without ADDRESS_ID
            matched_spr_ids = set()
            
            if 'ADDRESS_ID' in spr_processed.columns:
                # If SPR has ADDRESS_ID column, use it
                matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique())
                matched_spr_ids.discard('')  # Remove empty strings
                unmatched_mask = ~spr_processed['ADDRESS_ID'].isin(matched_spr_ids)
            else:
                # If no ADDRESS_ID column, matches are likely using pandas index
                # Let's check if ADDRESS_ID_SPR contains actual IDs or empty strings
                spr_ids_in_matches = matches_df['ADDRESS_ID_SPR'].unique()
                print(f"DEBUG: Sample ADDRESS_ID_SPR values: {spr_ids_in_matches[:10]}")
                
                if all(id_val == '' for id_val in spr_ids_in_matches):
                    # ADDRESS_ID_SPR is empty, matching was done on index
                    # We need to reconstruct which records were matched
                    # Use FULL_ADDRESS_SPR to identify matched records
                    matched_full_addresses = set(matches_df['FULL_ADDRESS_SPR'].unique())
                    unmatched_mask = ~spr_processed['FULL_ADDRESS'].isin(matched_full_addresses)
                else:
                    # ADDRESS_ID_SPR has actual values, use them
                    matched_spr_ids = set(spr_ids_in_matches)
                    matched_spr_ids.discard('')  # Remove empty strings
                    unmatched_mask = ~spr_processed.index.isin(matched_spr_ids)
            
            unmatched_spr = spr_processed[unmatched_mask].copy()
            print(f"DEBUG: After filtering, unmatched_spr has {len(unmatched_spr)} rows")
        
        # Ensure all required columns exist before adding analysis columns
        if 'COMPLETENESS_SCORE' not in unmatched_spr.columns:
            unmatched_spr['COMPLETENESS_SCORE'] = 0.0
        if 'STREET_NAME' not in unmatched_spr.columns:
            unmatched_spr['STREET_NAME'] = ''
        if 'HOUSE' not in unmatched_spr.columns:
            unmatched_spr['HOUSE'] = ''
        if 'BUILDING' not in unmatched_spr.columns:
            unmatched_spr['BUILDING'] = ''
        
        # Add analysis columns with safe operations
        unmatched_spr['IS_COMPLETE'] = unmatched_spr['COMPLETENESS_SCORE'].fillna(0) >= 0.8
        unmatched_spr['HAS_STREET'] = unmatched_spr['STREET_NAME'].notna() & (unmatched_spr['STREET_NAME'].astype(str) != '')
        unmatched_spr['HAS_HOUSE'] = unmatched_spr['HOUSE'].notna() & (unmatched_spr['HOUSE'].astype(str) != '')
        unmatched_spr['HAS_BUILDING'] = unmatched_spr['BUILDING'].notna() & (unmatched_spr['BUILDING'].astype(str) != '')
        
        return unmatched_spr


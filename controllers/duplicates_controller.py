import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.duplicate_models import SPRDuplicateDetector, DuplicateDataProcessor
from views.duplicates_view import (
    render_duplicates_overview,
    render_duplicates_filters,
    render_duplicates_table,
    render_duplicates_analysis,
    render_resolution_suggestions,
    render_duplicates_help,
    create_duplicates_overview_chart
)

logger = logging.getLogger(__name__)


class DuplicatesController:
    """
    Controller for managing SPR duplicates detection and analysis
    """
    
    def __init__(self):
        self.initialize_session_state()
        self.detector = None
        self.processor = DuplicateDataProcessor()
    
    def initialize_session_state(self):
        """Initialize session state variables for duplicates"""
        if "duplicates_results" not in st.session_state:
            st.session_state.duplicates_results = None
        if "duplicates_detector" not in st.session_state:
            st.session_state.duplicates_detector = None
        if "duplicates_analysis_complete" not in st.session_state:
            st.session_state.duplicates_analysis_complete = False
    
    def load_spr_data(self) -> Optional[pd.DataFrame]:
        """Load SPR data from session state or file"""
        # Try to get from session state first
        if 'spr_processed' in st.session_state and st.session_state.spr_processed is not None:
            return st.session_state.spr_processed
        
        # Try to load from file
        try:
            from models.address_models import load_registry_data_from_csv
            spr_df = load_registry_data_from_csv("SPR", "data/spr.csv")
            if not spr_df.empty:
                st.session_state.spr_processed = spr_df
                return spr_df
        except Exception as e:
            st.error(f"Error loading SPR data: {e}")
        
        return None
    
    def detect_duplicates(self, spr_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicates in SPR data"""
        try:
            with st.spinner("🔍 Detecting duplicates in SPR data..."):
                # Create detector
                self.detector = SPRDuplicateDetector(spr_df)
                
                # Detect duplicates
                results = self.detector.detect_duplicates()
                
                # Store results in session state
                st.session_state.duplicates_results = results
                st.session_state.duplicates_detector = self.detector
                st.session_state.duplicates_analysis_complete = True
                
                # Log results
                stats = results.get('duplicate_stats', {})
                logger.info(f"Duplicate detection completed: {stats.get('total_duplicates', 0)} duplicates found")
                
                return results
                
        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}")
            st.error(f"Error detecting duplicates: {e}")
            return {}
    
    def get_duplicate_results(self) -> Optional[Dict[str, Any]]:
        """Get current duplicate detection results"""
        return st.session_state.duplicates_results
    
    def get_detector(self) -> Optional[SPRDuplicateDetector]:
        """Get current duplicate detector instance"""
        return st.session_state.duplicates_detector
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze duplicate patterns"""
        detector = self.get_detector()
        if not detector:
            return {}
        
        try:
            return detector.analyze_duplicate_patterns()
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            st.error(f"Error analyzing patterns: {e}")
            return {}
    
    def get_resolution_suggestions(self) -> List[Dict]:
        """Get resolution suggestions"""
        detector = self.get_detector()
        if not detector:
            return []
        
        try:
            return detector.get_duplicate_resolution_suggestions()
        except Exception as e:
            logger.error(f"Error getting resolution suggestions: {e}")
            st.error(f"Error getting resolution suggestions: {e}")
            return []
    
    
    def reset_analysis(self):
        """Reset duplicate analysis"""
        st.session_state.duplicates_results = None
        st.session_state.duplicates_detector = None
        st.session_state.duplicates_analysis_complete = False
        self.detector = None
    
    def render_duplicates_tab(self):
        """Render the complete duplicates analysis tab"""
        st.header("🔍 SPR Address Duplicates Analysis")
        
        # Help section
        render_duplicates_help()
        
        # Load SPR data
        spr_df = self.load_spr_data()
        
        if spr_df is None or spr_df.empty:
            st.error("No SPR data available. Please load data in the Data Overview tab first.")
            return
        
        # Control buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 Detect Duplicates", type="primary", use_container_width=True):
                results = self.detect_duplicates(spr_df)
                if results:
                    st.success(f"✅ Duplicate detection completed! Found {results.get('duplicate_stats', {}).get('total_duplicates', 0)} duplicates.")
        
        with col2:
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                if st.session_state.duplicates_analysis_complete:
                    results = self.get_duplicate_results()
                    if results:
                        st.success("✅ Analysis refreshed!")
                else:
                    st.warning("Please detect duplicates first.")
        
        with col3:
            if st.button("🗑️ Reset", use_container_width=True):
                self.reset_analysis()
                st.success("✅ Analysis reset!")
                st.rerun()
        
        # Show results if available
        if st.session_state.duplicates_analysis_complete:
            self.render_analysis_results()
        else:
            st.info("👆 Click 'Detect Duplicates' to start analyzing SPR address duplicates.")
    
    def render_analysis_results(self):
        """Render the analysis results"""
        results = self.get_duplicate_results()
        
        if not results:
            st.error("No duplicate analysis results available.")
            return
        
        duplicate_stats = results.get('duplicate_stats', {})
        duplicate_groups = results.get('duplicate_groups', {})
        
        # Overview section
        render_duplicates_overview(duplicate_stats)
        
        # Overview chart
        overview_chart = create_duplicates_overview_chart(duplicate_stats)
        if overview_chart:
            st.plotly_chart(overview_chart, use_container_width=True)
        
        # Only show detailed analysis if duplicates exist
        if duplicate_stats.get('total_duplicates', 0) > 0:
            # Filters section
            filters = render_duplicates_filters()
            
            # Duplicates table
            duplicates_df = render_duplicates_table(duplicate_groups, filters)
            
            # Analysis section
            if duplicates_df is not None and not duplicates_df.empty:
                # Patterns analysis
                patterns = self.analyze_patterns()
                render_duplicates_analysis(patterns)
                
                # Resolution suggestions
                suggestions = self.get_resolution_suggestions()
                render_resolution_suggestions(suggestions)
                
            
        else:
            st.success("🎉 No duplicate addresses found in the SPR registry!")
            st.info("This means all addresses in the SPR registry are unique, which indicates good data quality.")
    
    def get_duplicate_summary_for_main_app(self) -> Dict[str, Any]:
        """Get duplicate summary for display in main app"""
        results = self.get_duplicate_results()
        
        if not results:
            return {
                'has_duplicates': False,
                'total_duplicates': 0,
                'duplicate_groups': 0,
                'duplicate_rate': 0.0
            }
        
        stats = results.get('duplicate_stats', {})
        return {
            'has_duplicates': stats.get('total_duplicates', 0) > 0,
            'total_duplicates': stats.get('total_duplicates', 0),
            'duplicate_groups': stats.get('unique_duplicate_groups', 0),
            'duplicate_rate': stats.get('duplicate_rate', 0.0),
            'largest_group': stats.get('largest_duplicate_group', 0)
        }
    
    
    def get_filtered_duplicates(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get filtered duplicates based on criteria"""
        results = self.get_duplicate_results()
        
        if not results:
            return {}
        
        duplicate_groups = results.get('duplicate_groups', {})
        filtered_groups = self.processor.filter_duplicates_by_criteria(
            duplicate_groups,
            min_count=filters.get('min_count', 2),
            max_count=filters.get('max_count', None),
            street_filter=filters.get('street_filter', None)
        )
        
        return filtered_groups
    
    def is_analysis_complete(self) -> bool:
        """Check if duplicate analysis is complete"""
        return st.session_state.duplicates_analysis_complete
    
    def get_duplicate_count(self) -> int:
        """Get total number of duplicates"""
        results = self.get_duplicate_results()
        if results:
            return results.get('duplicate_stats', {}).get('total_duplicates', 0)
        return 0
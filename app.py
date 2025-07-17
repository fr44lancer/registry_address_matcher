import streamlit as st
import pandas as pd
import time
from datetime import datetime
import logging

# Import our modules
from config.settings import APP_CONFIG, DB_CONFIG, MATCHING_CONFIG
from src.database.connection import get_database_manager
from src.database.data_loader import load_registry_data_cached
from src.matching.normalizer import AddressNormalizer
from src.matching.engine import AdvancedAddressMatcher
from src.quality.analyzer import DataQualityAnalyzer
from src.visualization.charts import VisualizationManager
from src.utils.export import ExportManager
from src.utils.logging import setup_logging, get_logger

# Setup logging
logger = setup_logging()
app_logger = get_logger('app')

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG.page_title,
    page_icon=APP_CONFIG.page_icon,
    layout=APP_CONFIG.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
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


def initialize_session_state():
    """Initialize session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'spr_data' not in st.session_state:
        st.session_state.spr_data = None
    if 'cad_data' not in st.session_state:
        st.session_state.cad_data = None
    if 'matches_data' not in st.session_state:
        st.session_state.matches_data = None
    if 'quality_metrics' not in st.session_state:
        st.session_state.quality_metrics = None


def preprocess_registries(spr_df, cad_df):
    """Comprehensive data preprocessing pipeline matching original address.py"""
    normalizer = AddressNormalizer()
    
    def process_registry(df, registry_name):
        """Process individual registry with comprehensive normalization"""
        processed = df.copy()
        
        # Handle missing values
        processed['STREET_NAME'] = processed['STREET_NAME'].fillna('')
        processed['HOUSE'] = processed['HOUSE'].fillna('')
        processed['BUILDING'] = processed['BUILDING'].fillna('')
        
        # Normalize fields
        processed['STREET_NORM'] = processed['STREET_NAME'].apply(normalizer.normalize_street_name)
        processed['HOUSE_NORM'] = processed['HOUSE'].apply(normalizer.normalize_house_number)
        processed['BUILDING_NORM'] = processed['BUILDING'].apply(normalizer.normalize_building_number)
        
        # Create composite addresses
        processed['FULL_ADDRESS'] = (
            processed['STREET_NORM'] + " " +
            processed['HOUSE_NORM'] + " " +
            processed['BUILDING_NORM']
        ).str.strip()
        
        # Create search keys
        processed['SEARCH_KEY'] = (
            processed['STREET_NORM'] + "_" + processed['HOUSE_NORM']
        )
        
        # Data quality metrics
        processed['COMPLETENESS_SCORE'] = (
            processed['STREET_NAME'].notna().astype(int) +
            processed['HOUSE'].notna().astype(int) +
            processed['BUILDING'].notna().astype(int)
        ) / 3
        
        app_logger.info(f"Processed {len(processed)} records for {registry_name}")
        return processed
    
    # Process both registries
    spr_processed = process_registry(spr_df, 'SPR')
    cad_processed = process_registry(cad_df, 'CAD')
    
    return spr_processed, cad_processed


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏘️ Address Registry Matcher</h1>
        <p>Advanced fuzzy matching system for address registries with comprehensive quality analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>⚙️ Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Database connection
    st.sidebar.subheader("Database Connection")
    
    if st.sidebar.button("Connect to Database"):
        with st.spinner("Connecting to database..."):
            st.session_state.db_manager = get_database_manager()
            if st.session_state.db_manager and st.session_state.db_manager.test_connection():
                st.sidebar.success("✅ Database connected successfully!")
                app_logger.info("Database connection established")
            else:
                st.sidebar.error("❌ Database connection failed!")
                app_logger.error("Database connection failed")
    
    # Data loading section
    if st.session_state.db_manager:
        st.sidebar.subheader("Data Loading")
        
        # Table selection
        spr_table = st.sidebar.text_input("SPR Table Name", value="spr")
        cad_table = st.sidebar.text_input("CAD Table Name", value="cadastre")
        
        if st.sidebar.button("Load Data"):
            with st.spinner("Loading registry data..."):
                # Load SPR data
                st.session_state.spr_data = load_registry_data_cached(
                    "SPR", spr_table, st.session_state.db_manager
                )
                
                # Load CAD data
                st.session_state.cad_data = load_registry_data_cached(
                    "CAD", cad_table, st.session_state.db_manager
                )
                
                if st.session_state.spr_data is not None and st.session_state.cad_data is not None:
                    st.sidebar.success("✅ Data loaded successfully!")
                    app_logger.info(f"Loaded {len(st.session_state.spr_data)} SPR and {len(st.session_state.cad_data)} CAD records")
                else:
                    st.sidebar.error("❌ Data loading failed!")
    
    # Matching configuration
    if st.session_state.spr_data is not None and st.session_state.cad_data is not None:
        st.sidebar.subheader("Matching Configuration")
        
        max_records = st.sidebar.slider(
            "Max SPR Records to Process",
            min_value=100,
            max_value=min(10000, len(st.session_state.spr_data)),
            value=min(1000, len(st.session_state.spr_data)),
            step=100
        )
        
        # Matching thresholds
        threshold_excellent = st.sidebar.slider(
            "Excellent Match Threshold",
            min_value=80.0,
            max_value=100.0,
            value=MATCHING_CONFIG.threshold_excellent,
            step=1.0
        )
        
        threshold_good = st.sidebar.slider(
            "Good Match Threshold",
            min_value=60.0,
            max_value=90.0,
            value=MATCHING_CONFIG.threshold_good,
            step=1.0
        )
        
        if st.sidebar.button("Start Matching"):
            with st.spinner("Processing and matching addresses..."):
                # Preprocess data
                spr_processed, cad_processed = preprocess_registries(
                    st.session_state.spr_data, st.session_state.cad_data
                )
                
                # Create matcher
                matcher = AdvancedAddressMatcher(
                    spr_processed, cad_processed, max_records=max_records
                )
                
                # Perform matching
                st.session_state.matches_data = matcher.match_addresses()
                
                # Analyze quality
                quality_analyzer = DataQualityAnalyzer()
                spr_quality = quality_analyzer.analyze_registry_quality(spr_processed, "SPR")
                cad_quality = quality_analyzer.analyze_registry_quality(cad_processed, "CAD")
                
                st.session_state.quality_metrics = {
                    'spr_quality': spr_quality,
                    'cad_quality': cad_quality,
                    'matching_stats': matcher.get_matching_statistics()
                }
                
                st.sidebar.success("✅ Matching completed!")
                app_logger.info(f"Matching completed with {len(st.session_state.matches_data)} matches")
    
    # Main content area
    if st.session_state.matches_data is not None:
        # Display results
        st.subheader("📊 Matching Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(st.session_state.matches_data))
        
        with col2:
            avg_score = st.session_state.matches_data['match_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        
        with col3:
            if len(st.session_state.spr_data) > 0:
                match_rate = len(st.session_state.matches_data) / len(st.session_state.spr_data) * 100
                st.metric("Match Rate", f"{match_rate:.1f}%")
            else:
                st.metric("Match Rate", "N/A")
        
        with col4:
            excellent_matches = len(st.session_state.matches_data[
                st.session_state.matches_data['match_score'] >= threshold_excellent
            ])
            st.metric("Excellent Matches", excellent_matches)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizations", "📋 Data Tables", "📊 Quality Analysis", "📥 Export"])
        
        with tab1:
            # Visualizations
            viz_manager = VisualizationManager()
            
            # Match quality chart
            quality_chart = viz_manager.create_match_quality_chart(st.session_state.matches_data)
            if quality_chart:
                st.plotly_chart(quality_chart, use_container_width=True)
            
            # Data quality comparison
            if st.session_state.quality_metrics:
                quality_dashboard = viz_manager.create_data_quality_dashboard(
                    st.session_state.quality_metrics['spr_quality'],
                    st.session_state.quality_metrics['cad_quality']
                )
                st.plotly_chart(quality_dashboard, use_container_width=True)
        
        with tab2:
            # Data tables
            st.subheader("Matched Addresses")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                quality_filter = st.selectbox(
                    "Filter by Quality",
                    ["All", "Excellent", "Good", "Poor"]
                )
            
            with col2:
                score_filter = st.slider(
                    "Minimum Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0
                )
            
            # Apply filters
            filtered_matches = st.session_state.matches_data.copy()
            
            if quality_filter != "All":
                filtered_matches = filtered_matches[
                    filtered_matches['match_quality'] == quality_filter
                ]
            
            filtered_matches = filtered_matches[
                filtered_matches['match_score'] >= score_filter
            ]
            
            st.dataframe(filtered_matches, use_container_width=True)
        
        with tab3:
            # Quality analysis
            st.subheader("Data Quality Analysis")
            
            if st.session_state.quality_metrics:
                quality_analyzer = DataQualityAnalyzer()
                
                # Display quality metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("SPR Registry Quality")
                    spr_quality = st.session_state.quality_metrics['spr_quality']
                    st.metric("Overall Quality Score", f"{spr_quality['overall_quality_score']:.1%}")
                    st.metric("Total Records", f"{spr_quality['total_records']:,}")
                    st.metric("Average Completeness", f"{spr_quality['completeness_metrics'].get('avg_completeness', 0):.1%}")
                
                with col2:
                    st.subheader("CAD Registry Quality")
                    cad_quality = st.session_state.quality_metrics['cad_quality']
                    st.metric("Overall Quality Score", f"{cad_quality['overall_quality_score']:.1%}")
                    st.metric("Total Records", f"{cad_quality['total_records']:,}")
                    st.metric("Average Completeness", f"{cad_quality['completeness_metrics'].get('avg_completeness', 0):.1%}")
                
                # Recommendations
                recommendations = quality_analyzer._generate_recommendations([spr_quality, cad_quality])
                if recommendations:
                    st.subheader("🔧 Recommendations")
                    for rec in recommendations:
                        st.warning(rec)
        
        with tab4:
            # Export functionality
            st.subheader("Export Results")
            
            export_manager = ExportManager()
            
            # Validate export data
            validation = export_manager.validate_export_data(
                st.session_state.matches_data,
                st.session_state.spr_data,
                st.session_state.cad_data
            )
            
            if validation['is_valid']:
                if st.button("Generate Export Package"):
                    with st.spinner("Creating export package..."):
                        try:
                            export_data = export_manager.create_export_package(
                                st.session_state.matches_data,
                                st.session_state.spr_data,
                                st.session_state.cad_data,
                                st.session_state.quality_metrics
                            )
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"address_matching_results_{timestamp}.zip"
                            
                            st.download_button(
                                label="📥 Download Export Package",
                                data=export_data,
                                file_name=filename,
                                mime="application/zip"
                            )
                            
                            st.success("✅ Export package created successfully!")
                            app_logger.info(f"Export package created: {filename}")
                            
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
                            app_logger.error(f"Export failed: {str(e)}")
            else:
                st.error("❌ Export validation failed:")
                for error in validation['errors']:
                    st.error(f"• {error}")
                for warning in validation['warnings']:
                    st.warning(f"• {warning}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Address Registry Matcher v1.0 - Advanced fuzzy matching with quality analysis*")


if __name__ == "__main__":
    main()
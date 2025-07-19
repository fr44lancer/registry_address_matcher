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


def clean_cad_address_prefix(df, column_name):
    """Remove geographic prefix from CAD address column"""
    if column_name in df.columns:
        prefix_to_remove = "Մարզ Շիրակ, համայնք Գյումրի, քաղաք Գյումրի"
        df[column_name] = df[column_name].astype(str).str.replace(
            prefix_to_remove, '', regex=False
        ).str.strip().str.lstrip(',').str.strip()
    return df


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Address Registry Matcher",
        page_icon="🏘️",
        layout="wide"
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
            spr_processed[['ADDRESS_ID', 'STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))

    with col2:
        st.subheader("Cadastre Sample Data")
        st.dataframe(
            cad_processed[['ADDRESS_ID', 'STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))


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







def render_interactive_match_explorer(matches_df):
    """Render interactive match explorer"""
    st.subheader("Interactive Match Explorer")

    # Simplified filters
    col1, col2, col3 = st.columns(3)

    with col1:
        street_filter = st.text_input("Filter by Street Name", "")

    with col2:
        if 'ORIGINAL_FULL_ADDRESS_SPR' in matches_df.columns:
            address_filter = st.text_input("Filter by ORIGINAL_FULL_ADDRESS_SPR", "")
        else:
            address_filter = ""
            st.info("ORIGINAL_FULL_ADDRESS_SPR not available")

    with col3:
        st.write("")  # Empty column for spacing

    # Apply filters
    filtered_matches = matches_df.copy()

    if street_filter:
        filtered_matches = filtered_matches[
            filtered_matches['STREET_NAME_SPR'].str.contains(street_filter, case=False, na=False) |
            filtered_matches['STREET_NAME_CAD'].str.contains(street_filter, case=False, na=False)
        ]
    
    if address_filter and 'ORIGINAL_FULL_ADDRESS_SPR' in matches_df.columns:
        filtered_matches = filtered_matches[
            filtered_matches['ORIGINAL_FULL_ADDRESS_SPR'].str.contains(address_filter, case=False, na=False)
        ]

    # Display filtered results
    st.subheader(f"Filtered Results ({len(filtered_matches)} matches)")

    # Check if there are any results to display
    if len(filtered_matches) == 0:
        st.info("No matches found with the current filters")
        return filtered_matches

    # Pagination
    page_size = 50
    total_pages = max(1, (len(filtered_matches) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

    start_idx = page * page_size
    end_idx = start_idx + page_size

    display_matches = filtered_matches.iloc[start_idx:end_idx]

    # Select only the requested columns
    requested_columns = [
        'ADDRESS_ID_SPR', 'ADDRESS_ID_CAD',
        'ORIGINAL_STREET_NAME_SPR', 'ORIGINAL_STREET_NAME_CAD',
        'ORIGINAL_FULL_ADDRESS_SPR', 'ORIGINAL_FULL_ADDRESS_CAD', 
        'NORMALIZED_FULL_ADDRESS_SPR', 'NORMALIZED_FULL_ADDRESS_CAD'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in requested_columns if col in display_matches.columns]
    
    if available_columns:
        display_matches = display_matches[available_columns].copy()
        
        # Clean up ORIGINAL_FULL_ADDRESS_CAD by removing the prefix
        display_matches = clean_cad_address_prefix(display_matches, 'ORIGINAL_FULL_ADDRESS_CAD')
    else:
        st.warning("⚠️ The requested columns are not available in the match results. Available columns: " + 
                  ", ".join(display_matches.columns.tolist()))

    st.dataframe(display_matches, use_container_width=True)

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




















def render_combined_unmatched_summary(unmatched_spr_df, unmatched_cad_df):
    """Render combined summary for both SPR and Cadastre unmatched addresses"""
    st.subheader("📊 Unmatched Addresses Summary")
    
    # Create columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 SPR Unmatched")
        if unmatched_spr_df is not None and not unmatched_spr_df.empty:
            spr_col1, spr_col2 = st.columns(2)
            with spr_col1:
                st.metric("Total Unmatched", f"{len(unmatched_spr_df):,}")
            with spr_col2:
                if 'IS_COMPLETE' in unmatched_spr_df.columns:
                    complete_count = unmatched_spr_df['IS_COMPLETE'].sum()
                    complete_pct = complete_count / len(unmatched_spr_df) * 100 if len(unmatched_spr_df) > 0 else 0
                    st.metric("Complete Records", f"{complete_count:,} ({complete_pct:.1f}%)")
                else:
                    st.metric("Complete Records", "N/A")
        else:
            st.info("No SPR unmatched addresses")
    
    with col2:
        st.markdown("### 🏛️ Cadastre Unmatched")
        if unmatched_cad_df is not None and not unmatched_cad_df.empty:
            cad_col1, cad_col2 = st.columns(2)
            with cad_col1:
                st.metric("Total Unmatched", f"{len(unmatched_cad_df):,}")
            with cad_col2:
                if 'IS_COMPLETE' in unmatched_cad_df.columns:
                    complete_count = unmatched_cad_df['IS_COMPLETE'].sum()
                    complete_pct = complete_count / len(unmatched_cad_df) * 100 if len(unmatched_cad_df) > 0 else 0
                    st.metric("Complete Records", f"{complete_count:,} ({complete_pct:.1f}%)")
                else:
                    st.metric("Complete Records", "N/A")
        else:
            st.info("No Cadastre unmatched addresses")


def render_registry_selector():
    """Render registry selector for filtering"""
    st.subheader("🔧 Registry Selection")
    
    registry_options = ["Both", "SPR Only", "Cadastre Only"]
    selected_registry = st.selectbox(
        "Select Registry to View",
        options=registry_options,
        index=0,
        help="Choose which registry's unmatched addresses to display"
    )
    
    return selected_registry


def render_unmatched_addresses_table_combined(spr_df, cad_df, registry_filter):
    """Render combined unmatched addresses table with registry filtering and search"""
    
    # Prepare data based on registry filter
    if registry_filter == "SPR Only":
        display_data = [(spr_df, "SPR", "📋")]
    elif registry_filter == "Cadastre Only":
        display_data = [(cad_df, "Cadastre", "🏛️")]
    else:  # Both
        display_data = []
        if spr_df is not None and not spr_df.empty:
            display_data.append((spr_df, "SPR", "📋"))
        if cad_df is not None and not cad_df.empty:
            display_data.append((cad_df, "Cadastre", "🏛️"))
    
    if not display_data:
        st.info("No unmatched addresses to display")
        return
    
    # Add filters section
    st.subheader("🔧 Filters & Search")
    
    # Create filter controls
    col2, col3 = st.columns(2)

    
    with col2:
        # Missing data filter
        missing_data_filter = st.selectbox(
            "Missing Data Filter",
            options=["All Records", "Missing Street", "Missing House", "Missing Building", "Complete Records Only"],
            key="combined_missing_filter"
        )
    
    with col3:
        # Street name search
        street_search = st.text_input("Search Street Name", "", key="combined_street_search")
    
    # Apply filters to each dataset
    def apply_filters(df):
        if df is None or df.empty:
            return df
        
        filtered_df = df.copy()
        
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
    
    # Apply filters to data
    filtered_display_data = []
    for df, registry_name, icon in display_data:
        filtered_df = apply_filters(df)
        if filtered_df is not None and not filtered_df.empty:
            filtered_display_data.append((filtered_df, registry_name, icon))
    
    if not filtered_display_data:
        st.info("No unmatched addresses match the current filters")
        return
    
    display_data = filtered_display_data
    
    # Display each registry's data
    for df, registry_name, icon in display_data:
        if df is None or df.empty:
            continue
            
        st.subheader(f"{icon} {registry_name} Unmatched Addresses ({len(df):,} records)")
        
        # Add helpful explanation
        with st.expander("ℹ️ Column Explanation", expanded=False):
            st.markdown("""
            **Original Data**: Shows data as it appears in the source database
            - `STREET_NAME`, `HOUSE`, `BUILDING`: Original values from the database
            
            **Normalized Data**: Shows how the address was processed for matching
            - `STREET_NORM`, `HOUSE_NORM`, `BUILDING_NORM`: Normalized individual components
            - `FULL_ADDRESS`: Complete normalized address string used for comparison
            
            The normalized values show exactly what string was compared against the other database.
            """)
        
        # Pagination for this registry
        page_size = 25  # Smaller page size when showing multiple registries
        total_pages = (len(df) - 1) // page_size + 1
        
        col1, col2 = st.columns([3, 1])
        with col1:
            page_key = f"page_{registry_name.lower()}"
            page = st.number_input(
                f"Page ({registry_name})", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                key=page_key
            ) - 1
        with col2:
            st.metric("Total Pages", total_pages)
        
        # Calculate pagination
        start_idx = page * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]
        
        # Select columns for display based on what's available
        base_columns = ['ADDRESS_ID', 'STREET_NAME', 'HOUSE', 'BUILDING']
        normalized_columns = ['STREET_NORM', 'HOUSE_NORM', 'BUILDING_NORM', 'FULL_ADDRESS']
        analysis_columns = ['COMPLETENESS_SCORE', 'HAS_STREET', 'HAS_HOUSE', 'HAS_BUILDING']
        
        display_columns = []
        
        # Add base columns (original data)
        for col in base_columns:
            if col in page_df.columns:
                display_columns.append(col)
        
        # Add normalized columns (what was actually compared)
        for col in normalized_columns:
            if col in page_df.columns:
                display_columns.append(col)
        
        # Add analysis columns if available
        for col in analysis_columns:
            if col in page_df.columns:
                display_columns.append(col)
        
        if not display_columns:
            st.error(f"No displayable columns found for {registry_name}")
            continue
        
        # Create display dataframe
        display_df = page_df[display_columns].copy()
        
        # Rename columns to be more descriptive
        column_renames = {
            'STREET_NAME': 'Street (Original)',
            'HOUSE': 'House (Original)', 
            'BUILDING': 'Building (Original)',
            'STREET_NORM': 'Street (Normalized)',
            'HOUSE_NORM': 'House (Normalized)',
            'BUILDING_NORM': 'Building (Normalized)', 
            'FULL_ADDRESS': 'Full Address (Normalized)',
            'COMPLETENESS_SCORE': 'Completeness',
            'HAS_STREET': 'Has Street',
            'HAS_HOUSE': 'Has House',
            'HAS_BUILDING': 'Has Building'
        }
        
        # Apply renames only for columns that exist
        renames_to_apply = {old: new for old, new in column_renames.items() if old in display_df.columns}
        display_df = display_df.rename(columns=renames_to_apply)
        
        # Format completeness score as percentage if available
        if 'Completeness' in display_df.columns:
            display_df['Completeness'] = display_df['Completeness'].apply(lambda x: f"{x:.1%}")
        
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
        if 'Completeness' in display_df.columns:
            styled_df = styled_df.applymap(highlight_completeness, subset=['Completeness'])
        
        # Apply boolean highlighting if columns exist
        boolean_columns = [col for col in ['Has Street', 'Has House', 'Has Building'] if col in display_df.columns]
        if boolean_columns:
            styled_df = styled_df.applymap(highlight_boolean, subset=boolean_columns)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Show pagination info
        st.info(f"Showing {registry_name} records {start_idx + 1:,} to {min(end_idx, len(df)):,} of {len(df):,}")
        
        # Add separator between registries
        if len(display_data) > 1 and registry_name != display_data[-1][1]:
            st.divider()





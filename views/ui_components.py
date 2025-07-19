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
        # Include SUB_STREET_NAME if it exists in the Cadastre data
        cad_columns = ['ADDRESS_ID', 'STREET_NAME']
        if 'SUB_STREET_NAME' in cad_processed.columns:
            cad_columns.append('SUB_STREET_NAME')
        cad_columns.extend(['HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE'])
        
        st.dataframe(cad_processed[cad_columns].head(10))


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
    """Enhanced manual review interface with detailed comparison tools"""
    st.subheader("🔍 Enhanced Manual Review Interface")

    if len(filtered_matches) == 0:
        st.info("No matches available for review.")
        return

    # Match selection with enhanced preview
    st.markdown("### Select Match to Review")
    
    # Initialize session state for navigation
    if 'manual_review_idx' not in st.session_state:
        st.session_state.manual_review_idx = 0
    
    # Ensure index is within bounds
    if st.session_state.manual_review_idx >= len(filtered_matches):
        st.session_state.manual_review_idx = len(filtered_matches) - 1
    if st.session_state.manual_review_idx < 0:
        st.session_state.manual_review_idx = 0
    
    col1, col2 = st.columns([3, 1])
    with col1:
        def format_match_option(x):
            match = filtered_matches.iloc[x]
            spr_addr = match.get('ORIGINAL_FULL_ADDRESS_SPR', 'N/A')
            cad_addr = match.get('ORIGINAL_FULL_ADDRESS_CAD', 'N/A')
            score = match.get('MATCH_SCORE', 0)
            
            # Truncate long addresses for dropdown readability
            max_length = 40
            if len(spr_addr) > max_length:
                spr_addr = spr_addr[:max_length] + "..."
            if len(cad_addr) > max_length:
                cad_addr = cad_addr[:max_length] + "..."
            
            return f"Match {x + 1}: {spr_addr} ↔ {cad_addr} (Score: {score})"
        
        review_idx = st.selectbox(
            "Choose match",
            range(len(filtered_matches)),
            format_func=format_match_option,
            index=st.session_state.manual_review_idx,
            key="manual_review_selectbox"
        )
        
        # Update session state when selectbox changes
        if review_idx != st.session_state.manual_review_idx:
            st.session_state.manual_review_idx = review_idx
    
    with col2:
        st.metric("Total Matches", len(filtered_matches))
        st.metric("Current", f"{review_idx + 1} of {len(filtered_matches)}")

    if review_idx is not None:
        review_match = filtered_matches.iloc[review_idx]
        
        # Match quality assessment
        score = review_match.get('MATCH_SCORE', 0)
        match_type = review_match.get('MATCH_TYPE', 'Unknown')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if score >= 95:
                st.success(f"🟢 Excellent Match: {score}")
            elif score >= 85:
                st.info(f"🔵 Good Match: {score}")
            elif score >= 70:
                st.warning(f"🟡 Fair Match: {score}")
            else:
                st.error(f"🔴 Poor Match: {score}")
        
        with col2:
            st.info(f"**Match Type:** {match_type}")
        
        with col3:
            completeness_spr = review_match.get('COMPLETENESS_SPR', 0)
            completeness_cad = review_match.get('COMPLETENESS_CAD', 0)
            avg_completeness = (completeness_spr + completeness_cad) / 2
            st.metric("Avg Completeness", f"{avg_completeness:.1%}")

        st.divider()

        # Detailed record comparison
        st.markdown("### 📋 Detailed Record Comparison")
        
        # Address IDs section
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            st.markdown("#### 📋 SPR Address ID")
            st.code(review_match.get('ADDRESS_ID_SPR', 'Not Available'), language=None)
        with id_col2:
            st.markdown("#### 🏛️ Cadastre Address ID")
            st.code(review_match.get('ADDRESS_ID_CAD', 'Not Available'), language=None)

        # Original vs Normalized comparison
        comparison_tabs = st.tabs(["📝 Original Data", "🔧 Normalized Data", "📊 Detailed Analysis"])
        
        with comparison_tabs[0]:
            st.markdown("#### Original Address Components")
            orig_col1, orig_col2 = st.columns(2)
            
            with orig_col1:
                st.markdown("**📋 SPR Original**")
                st.write(f"**Street:** {review_match.get('ORIGINAL_STREET_NAME_SPR', 'N/A')}")
                st.write(f"**House:** {review_match.get('HOUSE_SPR', 'N/A')}")
                st.write(f"**Building:** {review_match.get('BUILDING_SPR', 'N/A')}")
                st.markdown("**Full Original Address:**")
                st.code(review_match.get('ORIGINAL_FULL_ADDRESS_SPR', 'N/A'), language=None)
            
            with orig_col2:
                st.markdown("**🏛️ Cadastre Original**")
                st.write(f"**Street:** {review_match.get('ORIGINAL_STREET_NAME_CAD', 'N/A')}")
                sub_street = review_match.get('ORIGINAL_SUB_STREET_NAME_CAD', '')
                if sub_street:
                    st.write(f"**Sub-Street:** {sub_street}")
                st.write(f"**House:** {review_match.get('HOUSE_CAD', 'N/A')}")
                st.write(f"**Building:** {review_match.get('BUILDING_CAD', 'N/A')}")
                st.markdown("**Full Original Address:**")
                st.code(review_match.get('ORIGINAL_FULL_ADDRESS_CAD', 'N/A'), language=None)

        with comparison_tabs[1]:
            st.markdown("#### Normalized Address Components (Used for Matching)")
            norm_col1, norm_col2 = st.columns(2)
            
            with norm_col1:
                st.markdown("**📋 SPR Normalized**")
                st.write(f"**Street:** {review_match.get('STREET_NAME_SPR', 'N/A')}")
                st.write(f"**House:** {review_match.get('HOUSE_SPR', 'N/A')}")
                st.write(f"**Building:** {review_match.get('BUILDING_SPR', 'N/A')}")
                st.markdown("**Full Normalized Address:**")
                st.code(review_match.get('NORMALIZED_FULL_ADDRESS_SPR', review_match.get('FULL_ADDRESS_SPR', 'N/A')), language=None)
            
            with norm_col2:
                st.markdown("**🏛️ Cadastre Normalized**")
                st.write(f"**Street:** {review_match.get('STREET_NAME_CAD', 'N/A')}")
                sub_street_norm = review_match.get('SUB_STREET_NAME_CAD', '')
                if sub_street_norm:
                    st.write(f"**Sub-Street:** {sub_street_norm}")
                st.write(f"**House:** {review_match.get('HOUSE_CAD', 'N/A')}")
                st.write(f"**Building:** {review_match.get('BUILDING_CAD', 'N/A')}")
                st.markdown("**Full Normalized Address:**")
                st.code(review_match.get('NORMALIZED_FULL_ADDRESS_CAD', review_match.get('FULL_ADDRESS_CAD', 'N/A')), language=None)

        with comparison_tabs[2]:
            st.markdown("#### Match Analysis")
            
            # Component-by-component comparison
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("**🔍 Component Comparison**")
                
                # Street comparison
                spr_street = review_match.get('STREET_NAME_SPR', '')
                cad_street = review_match.get('STREET_NAME_CAD', '')
                if spr_street == cad_street:
                    st.success("✅ Streets match exactly")
                else:
                    st.warning("⚠️ Streets differ")
                    st.write(f"SPR: `{spr_street}`")
                    st.write(f"CAD: `{cad_street}`")
                
                # Sub-street comparison (Cadastre only)
                cad_sub_street = review_match.get('SUB_STREET_NAME_CAD', '')
                if cad_sub_street:
                    st.info("📍 Cadastre has sub-street")
                    st.write(f"CAD Sub-Street: `{cad_sub_street}`")
                else:
                    st.info("📍 No sub-street in Cadastre")
                
                # House comparison
                spr_house = str(review_match.get('HOUSE_SPR', ''))
                cad_house = str(review_match.get('HOUSE_CAD', ''))
                if spr_house == cad_house:
                    st.success("✅ House numbers match")
                else:
                    st.warning("⚠️ House numbers differ")
                    st.write(f"SPR: `{spr_house}`")
                    st.write(f"CAD: `{cad_house}`")
                
                # Building comparison
                spr_building = str(review_match.get('BUILDING_SPR', ''))
                cad_building = str(review_match.get('BUILDING_CAD', ''))
                if spr_building == cad_building:
                    st.success("✅ Building numbers match")
                else:
                    st.warning("⚠️ Building numbers differ")
                    st.write(f"SPR: `{spr_building}`")
                    st.write(f"CAD: `{cad_building}`")
            
            with analysis_col2:
                st.markdown("**📈 Quality Metrics**")
                st.write(f"**SPR Completeness:** {review_match.get('COMPLETENESS_SPR', 0):.1%}")
                st.write(f"**Cadastre Completeness:** {review_match.get('COMPLETENESS_CAD', 0):.1%}")
                st.write(f"**Match Timestamp:** {review_match.get('MATCH_TIMESTAMP', 'N/A')}")
                st.write(f"**Candidates Count:** {review_match.get('CANDIDATES_COUNT', 'N/A')}")
                
                # Character-level differences
                st.markdown("**🔤 Character Differences**")
                spr_full = review_match.get('NORMALIZED_FULL_ADDRESS_SPR', '')
                cad_full = review_match.get('NORMALIZED_FULL_ADDRESS_CAD', '')
                
                if spr_full == cad_full:
                    st.success("🎯 Perfect character match")
                else:
                    char_diff = abs(len(spr_full) - len(cad_full))
                    st.write(f"Length difference: {char_diff} characters")
                    
                    # Show first few character differences
                    min_len = min(len(spr_full), len(cad_full))
                    diff_count = sum(1 for i in range(min_len) if spr_full[i] != cad_full[i])
                    st.write(f"Character differences: {diff_count}")

        st.divider()

        # Navigation
        st.markdown("### 🧭 Navigation")
        st.caption("💡 Tip: Use the dropdown to jump to any match, or use the buttons below to navigate step by step")
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            if st.button("⬅️ Previous Match", disabled=(review_idx == 0), key="prev_match_btn"):
                st.session_state.manual_review_idx = max(0, review_idx - 1)
                st.rerun()
        
        with nav_col2:
            st.write(f"Match {review_idx + 1} of {len(filtered_matches)}")
        
        with nav_col3:
            if st.button("➡️ Next Match", disabled=(review_idx == len(filtered_matches) - 1), key="next_match_btn"):
                st.session_state.manual_review_idx = min(len(filtered_matches) - 1, review_idx + 1)
                st.rerun()




















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
            - `STREET_NAME`, `SUB_STREET_NAME` (Cadastre only), `HOUSE`, `BUILDING`: Original values from the database
            
            **Normalized Data**: Shows how the address was processed for matching
            - `STREET_NORM`, `SUB_STREET_NORM` (Cadastre only), `HOUSE_NORM`, `BUILDING_NORM`: Normalized individual components
            - `FULL_ADDRESS`: Complete normalized address string used for comparison (includes sub-street for Cadastre)
            
            The normalized values show exactly what string was compared against the other database.
            Note: Cadastre addresses may include a sub-street component between street and house number.
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
        base_columns = ['ADDRESS_ID', 'STREET_NAME']
        # Add SUB_STREET_NAME if available (for Cadastre)
        if 'SUB_STREET_NAME' in page_df.columns:
            base_columns.append('SUB_STREET_NAME')
        base_columns.extend(['HOUSE', 'BUILDING'])
        
        normalized_columns = ['STREET_NORM']
        # Add SUB_STREET_NORM if available (for Cadastre)
        if 'SUB_STREET_NORM' in page_df.columns:
            normalized_columns.append('SUB_STREET_NORM')
        normalized_columns.extend(['HOUSE_NORM', 'BUILDING_NORM', 'FULL_ADDRESS'])
        
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
            'SUB_STREET_NAME': 'Sub-Street (Original)',
            'HOUSE': 'House (Original)', 
            'BUILDING': 'Building (Original)',
            'STREET_NORM': 'Street (Normalized)',
            'SUB_STREET_NORM': 'Sub-Street (Normalized)',
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


def render_unmatched_street_names_tab(spr_missing_df, cad_missing_df):
    """Render unmatched street names comparison tab"""
    st.subheader("🛣️ Unmatched Street Names Comparison")
    
    # Summary metrics
    st.markdown("### 📊 Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spr_count = len(spr_missing_df) if spr_missing_df is not None else 0
        st.metric("Streets in SPR only", f"{spr_count:,}")
    
    with col2:
        cad_count = len(cad_missing_df) if cad_missing_df is not None else 0
        st.metric("Streets in Cadastre only", f"{cad_count:,}")
    
    with col3:
        total_unmatched = spr_count + cad_count
        st.metric("Total Unmatched", f"{total_unmatched:,}")
    
    if total_unmatched == 0:
        st.success("🎉 Perfect! All street names are present in both registries.")
        return
    
    # Add explanation
    with st.expander("ℹ️ What does this show?", expanded=False):
        st.markdown("""
        This tab compares **unique street names** between the SPR and Cadastre registries:
        
        - **Streets in SPR only**: Street names that exist in SPR but are missing from Cadastre
        - **Streets in Cadastre only**: Street names that exist in Cadastre but are missing from SPR
        
        These differences could be due to:
        - Different naming conventions between registries
        - Streets that exist in one registry but not the other
        - Data entry variations or typos
        - Historical name changes not reflected in both systems
        
        **Note**: This comparison uses normalized street names for accuracy.
        """)
    
    # Search functionality
    st.markdown("### 🔍 Search Street Names")
    search_term = st.text_input("Search for a street name:", "", key="street_search")
    
    # Display tables side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Streets in SPR only (missing in Cadastre)")
        
        if spr_missing_df is not None and not spr_missing_df.empty:
            # Apply search filter
            display_spr = spr_missing_df.copy()
            if search_term:
                mask = display_spr['STREET_NAME'].str.contains(search_term, case=False, na=False)
                display_spr = display_spr[mask]
            
            if not display_spr.empty:
                # Pagination for SPR
                page_size = 20
                total_pages = max(1, (len(display_spr) - 1) // page_size + 1)
                page_spr = st.number_input(
                    "Page (SPR)", 
                    min_value=1, 
                    max_value=total_pages, 
                    value=1,
                    key="page_spr_streets"
                ) - 1
                
                start_idx = page_spr * page_size
                end_idx = start_idx + page_size
                page_data = display_spr.iloc[start_idx:end_idx]
                
                st.dataframe(
                    page_data.reset_index(drop=True), 
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(display_spr))} of {len(display_spr)} streets")
            else:
                st.info("No streets found matching the search term")
        else:
            st.info("No streets found in SPR that are missing in Cadastre")
    
    with col2:
        st.markdown("#### 🏛️ Streets in Cadastre only (missing in SPR)")
        
        if cad_missing_df is not None and not cad_missing_df.empty:
            # Apply search filter
            display_cad = cad_missing_df.copy()
            if search_term:
                mask = display_cad['STREET_NAME'].str.contains(search_term, case=False, na=False)
                display_cad = display_cad[mask]
            
            if not display_cad.empty:
                # Pagination for Cadastre
                page_size = 20
                total_pages = max(1, (len(display_cad) - 1) // page_size + 1)
                page_cad = st.number_input(
                    "Page (Cadastre)", 
                    min_value=1, 
                    max_value=total_pages, 
                    value=1,
                    key="page_cad_streets"
                ) - 1
                
                start_idx = page_cad * page_size
                end_idx = start_idx + page_size
                page_data = display_cad.iloc[start_idx:end_idx]
                
                st.dataframe(
                    page_data.reset_index(drop=True), 
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(display_cad))} of {len(display_cad)} streets")
            else:
                st.info("No streets found matching the search term")
        else:
            st.info("No streets found in Cadastre that are missing in SPR")





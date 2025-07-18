import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime
import json


def render_duplicates_overview(duplicate_stats: Dict):
    """Render the duplicates overview section"""
    st.subheader("🔍 SPR Duplicates Overview")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records",
            f"{duplicate_stats.get('total_records', 0):,}",
            help="Total number of valid SPR records"
        )
    
    with col2:
        st.metric(
            "Duplicate Records",
            f"{duplicate_stats.get('total_duplicates', 0):,}",
            help="Total number of duplicate records found"
        )
    
    with col3:
        st.metric(
            "Duplicate Groups",
            f"{duplicate_stats.get('unique_duplicate_groups', 0):,}",
            help="Number of unique addresses with duplicates"
        )
    
    with col4:
        duplicate_rate = duplicate_stats.get('duplicate_rate', 0)
        st.metric(
            "Duplicate Rate",
            f"{duplicate_rate:.1%}",
            help="Percentage of records that are duplicates"
        )
    
    # Additional stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Empty Addresses",
            f"{duplicate_stats.get('empty_addresses', 0):,}",
            help="Records with empty or missing addresses"
        )
    
    with col2:
        st.metric(
            "Largest Group",
            f"{duplicate_stats.get('largest_duplicate_group', 0):,}",
            help="Size of the largest duplicate group"
        )
    
    with col3:
        # Calculate efficiency loss
        efficiency_loss = duplicate_stats.get('total_duplicates', 0) - duplicate_stats.get('unique_duplicate_groups', 0)
        st.metric(
            "Efficiency Loss",
            f"{efficiency_loss:,}",
            help="Extra records that could be removed"
        )


def render_duplicates_filters():
    """Render filtering controls for duplicates"""
    st.subheader("🎛️ Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_count = st.slider(
            "Minimum Group Size",
            min_value=2,
            max_value=20,
            value=2,
            help="Show only groups with at least this many duplicates"
        )
    
    with col2:
        max_count = st.slider(
            "Maximum Group Size",
            min_value=2,
            max_value=50,
            value=20,
            help="Show only groups with at most this many duplicates"
        )
    
    with col3:
        street_filter = st.text_input(
            "Street Name Filter",
            placeholder="Enter street name to filter",
            help="Filter duplicates by street name (case insensitive)"
        )
    
    return {
        'min_count': min_count,
        'max_count': max_count,
        'street_filter': street_filter if street_filter else None
    }


def render_duplicates_table(duplicate_groups: Dict, filters: Dict):
    """Render the main duplicates table"""
    st.subheader("📋 Duplicate Records")
    
    if not duplicate_groups:
        st.info("No duplicate records found with the current filters.")
        return None
    
    # Apply filters
    filtered_groups = apply_filters(duplicate_groups, filters)
    
    if not filtered_groups:
        st.warning("No duplicates match the current filter criteria.")
        return None
    
    # Create display DataFrame
    display_data = []
    for address, group in filtered_groups.items():
        for i, record in enumerate(group['records']):
            display_data.append({
                'Full Address': address,
                'Group Size': group['count'],
                'Record #': i + 1,
                'Street Name': record.get('STREET_NAME', ''),
                'House': record.get('HOUSE', ''),
                'Building': record.get('BUILDING', ''),
                'Address ID': record.get('ADDRESS_ID', ''),
                'Record Index': record.get('RECORD_INDEX', '')
            })
    
    df = pd.DataFrame(display_data)
    
    # Display summary
    st.info(f"Showing {len(filtered_groups)} duplicate groups with {len(df)} total records")
    
    # Pagination
    page_size = 50
    total_pages = (len(df) - 1) // page_size + 1
    
    if total_pages > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            help=f"Navigate through {total_pages} pages of results"
        ) - 1
        
        start_idx = page * page_size
        end_idx = start_idx + page_size
        df_page = df.iloc[start_idx:end_idx]
    else:
        df_page = df
    
    # Style the table
    def highlight_groups(val):
        """Highlight different duplicate groups"""
        if val is None:
            return ''
        # Use different colors for different group sizes
        group_size = df_page[df_page['Full Address'] == val]['Group Size'].iloc[0] if not df_page.empty else 0
        if group_size >= 10:
            return 'background-color: #ffcccc'  # Red for large groups
        elif group_size >= 5:
            return 'background-color: #fff3cd'  # Yellow for medium groups
        else:
            return 'background-color: #d4edda'  # Green for small groups
    
    # Apply styling
    styled_df = df_page.style.applymap(
        lambda x: highlight_groups(x) if x in df_page['Full Address'].values else '',
        subset=['Full Address']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    return df


def render_duplicates_analysis(patterns_analysis: Dict):
    """Render duplicates analysis charts"""
    st.subheader("📊 Duplicate Patterns Analysis")
    
    if not patterns_analysis:
        st.info("No patterns analysis available.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Group Size Distribution", "🛣️ Street Analysis", "🏠 House Patterns"])
    
    with tab1:
        render_group_size_distribution(patterns_analysis.get('count_distribution', {}))
    
    with tab2:
        render_street_analysis(patterns_analysis.get('top_streets_with_duplicates', {}))
    
    with tab3:
        render_house_patterns(patterns_analysis.get('top_house_patterns', {}))


def render_group_size_distribution(count_distribution: Dict):
    """Render group size distribution chart"""
    if not count_distribution:
        st.info("No group size distribution data available.")
        return
    
    # Create bar chart
    sizes = list(count_distribution.keys())
    counts = list(count_distribution.values())
    
    fig = px.bar(
        x=sizes,
        y=counts,
        title="Distribution of Duplicate Group Sizes",
        labels={'x': 'Group Size (Number of Duplicates)', 'y': 'Number of Groups'},
        color=counts,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis_title="Group Size",
        yaxis_title="Number of Groups",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_groups = sum(counts)
        st.metric("Total Groups", f"{total_groups:,}")
    
    with col2:
        avg_group_size = sum(size * count for size, count in count_distribution.items()) / sum(counts) if counts else 0
        st.metric("Average Group Size", f"{avg_group_size:.1f}")
    
    with col3:
        max_group_size = max(sizes) if sizes else 0
        st.metric("Max Group Size", f"{max_group_size:,}")


def render_street_analysis(top_streets: Dict):
    """Render street analysis chart"""
    if not top_streets:
        st.info("No street analysis data available.")
        return
    
    # Create horizontal bar chart
    streets = list(top_streets.keys())
    duplicate_counts = list(top_streets.values())
    
    fig = px.bar(
        x=duplicate_counts,
        y=streets,
        orientation='h',
        title="Top Streets with Most Duplicates",
        labels={'x': 'Number of Duplicate Records', 'y': 'Street Name'},
        color=duplicate_counts,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Number of Duplicate Records",
        yaxis_title="Street Name",
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Street summary
    st.write("**Top 5 Streets with Most Duplicates:**")
    for i, (street, count) in enumerate(list(top_streets.items())[:5], 1):
        st.write(f"{i}. **{street}**: {count} duplicate records")


def render_house_patterns(house_patterns: Dict):
    """Render house number patterns"""
    if not house_patterns:
        st.info("No house pattern data available.")
        return
    
    # Create pie chart for top house patterns
    houses = list(house_patterns.keys())[:10]  # Top 10
    counts = list(house_patterns.values())[:10]
    
    fig = px.pie(
        values=counts,
        names=houses,
        title="Top House Numbers in Duplicate Groups",
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # House patterns summary
    st.write("**Most Common House Numbers in Duplicates:**")
    for i, (house, count) in enumerate(list(house_patterns.items())[:5], 1):
        st.write(f"{i}. **{house}**: appears in {count} duplicate groups")


def render_resolution_suggestions(suggestions: List[Dict]):
    """Render resolution suggestions"""
    st.subheader("💡 Resolution Suggestions")
    
    if not suggestions:
        st.info("No resolution suggestions available.")
        return
    
    # Summary of suggestions
    merge_count = sum(1 for s in suggestions if s.get('suggestion_type') == 'merge')
    review_count = sum(1 for s in suggestions if s.get('suggestion_type') == 'review')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Ready to Merge", f"{merge_count:,}", help="Identical records that can be automatically merged")
    
    with col2:
        st.metric("Need Review", f"{review_count:,}", help="Records with differences that need manual review")
    
    # Detailed suggestions
    st.write("**Detailed Resolution Suggestions:**")
    
    # Group suggestions by type
    merge_suggestions = [s for s in suggestions if s.get('suggestion_type') == 'merge']
    review_suggestions = [s for s in suggestions if s.get('suggestion_type') == 'review']
    
    # Show merge suggestions
    if merge_suggestions:
        with st.expander(f"🔀 Ready to Merge ({len(merge_suggestions)} groups)", expanded=True):
            for i, suggestion in enumerate(merge_suggestions[:10], 1):  # Show first 10
                st.write(f"**{i}. {suggestion['address']}**")
                st.write(f"   - Duplicate count: {suggestion['duplicate_count']}")
                st.write(f"   - Action: {suggestion['suggestion_text']}")
                st.write("---")
    
    # Show review suggestions
    if review_suggestions:
        with st.expander(f"🔍 Need Review ({len(review_suggestions)} groups)", expanded=False):
            for i, suggestion in enumerate(review_suggestions[:10], 1):  # Show first 10
                st.write(f"**{i}. {suggestion['address']}**")
                st.write(f"   - Duplicate count: {suggestion['duplicate_count']}")
                st.write(f"   - Action: {suggestion['suggestion_text']}")
                st.write("---")


def render_duplicates_export(duplicates_report: Dict):
    """Render export options for duplicates"""
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary Report"):
            summary_data = {
                'summary': duplicates_report.get('summary', {}),
                'export_timestamp': datetime.now().isoformat()
            }
            
            json_data = json.dumps(summary_data, indent=2)
            st.download_button(
                label="Download Summary JSON",
                data=json_data,
                file_name=f"spr_duplicates_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📋 Export Detailed Report"):
            st.download_button(
                label="Download Detailed Report",
                data=json.dumps(duplicates_report, indent=2),
                file_name=f"spr_duplicates_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📊 Export Patterns Analysis"):
            patterns_data = duplicates_report.get('patterns_analysis', {})
            st.download_button(
                label="Download Patterns JSON",
                data=json.dumps(patterns_data, indent=2),
                file_name=f"spr_duplicates_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def apply_filters(duplicate_groups: Dict, filters: Dict) -> Dict:
    """Apply filters to duplicate groups"""
    filtered_groups = {}
    
    for address, group in duplicate_groups.items():
        # Filter by count
        if group['count'] < filters.get('min_count', 2):
            continue
        if group['count'] > filters.get('max_count', 50):
            continue
        
        # Filter by street name
        if filters.get('street_filter'):
            street_filter = filters['street_filter'].lower()
            has_street = any(
                street_filter in record.get('STREET_NAME', '').lower()
                for record in group['records']
            )
            if not has_street:
                continue
        
        filtered_groups[address] = group
    
    return filtered_groups


def render_duplicates_help():
    """Render help information for duplicates page"""
    with st.expander("ℹ️ Help - Understanding SPR Duplicates", expanded=False):
        st.markdown("""
        ### What are SPR Duplicates?
        
        SPR duplicates are records in the SPR registry that have identical full addresses 
        (combining street name, house number, and building number).
        
        ### Key Metrics Explained:
        
        - **Total Records**: Total number of valid SPR records with non-empty addresses
        - **Duplicate Records**: Total number of records that are duplicates of other records
        - **Duplicate Groups**: Number of unique addresses that have duplicates
        - **Duplicate Rate**: Percentage of records that are duplicates
        - **Efficiency Loss**: Extra records that could be removed (duplicates - unique groups)
        
        ### Resolution Types:
        
        - **🔀 Ready to Merge**: Identical records that can be automatically merged
        - **🔍 Need Review**: Records with differences that need manual review
        
        ### Color Coding:
        
        - 🟢 **Green**: Small duplicate groups (2-4 records)
        - 🟡 **Yellow**: Medium duplicate groups (5-9 records)  
        - 🔴 **Red**: Large duplicate groups (10+ records)
        
        ### Tips:
        
        - Use filters to focus on specific duplicate patterns
        - Large duplicate groups often indicate data quality issues
        - Export reports for detailed analysis and cleanup planning
        """)


def create_duplicates_overview_chart(duplicate_stats: Dict):
    """Create overview chart for duplicates"""
    if not duplicate_stats:
        return None
    
    # Create a pie chart showing the distribution
    labels = ['Unique Records', 'Duplicate Records']
    values = [
        duplicate_stats.get('total_records', 0) - duplicate_stats.get('total_duplicates', 0),
        duplicate_stats.get('total_duplicates', 0)
    ]
    
    fig = px.pie(
        values=values,
        names=labels,
        title="SPR Records Distribution",
        color_discrete_map={'Unique Records': '#2E8B57', 'Duplicate Records': '#DC143C'}
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig
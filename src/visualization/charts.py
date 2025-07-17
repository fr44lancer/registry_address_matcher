import logging
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Handles all visualization components for the address matcher"""
    
    def __init__(self):
        self.color_palette = {
            'excellent': '#28a745',
            'good': '#ffc107',
            'poor': '#dc3545',
            'primary': '#007bff',
            'secondary': '#6c757d'
        }
    
    def create_match_quality_chart(self, matches_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create comprehensive match quality visualization"""
        if matches_df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Match Score Distribution', 
                'Match Quality Distribution',
                'Match Type Analysis', 
                'Score vs Quality Correlation'
            ),
            specs=[
                [{"secondary_y": False}, {"type": "pie"}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Score distribution histogram
        fig.add_trace(
            go.Histogram(
                x=matches_df['match_score'], 
                nbinsx=20, 
                name='Score Distribution',
                marker_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # 2. Match quality pie chart
        if 'match_quality' in matches_df.columns:
            quality_counts = matches_df['match_quality'].value_counts()
            colors = [self.color_palette.get(q.lower(), self.color_palette['secondary']) 
                     for q in quality_counts.index]
            
            fig.add_trace(
                go.Pie(
                    labels=quality_counts.index, 
                    values=quality_counts.values, 
                    name='Match Quality',
                    marker_colors=colors
                ),
                row=1, col=2
            )
        
        # 3. Match type analysis
        if 'match_type' in matches_df.columns:
            type_counts = matches_df['match_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=type_counts.index, 
                    y=type_counts.values, 
                    name='Match Types',
                    marker_color=self.color_palette['secondary']
                ),
                row=2, col=1
            )
        
        # 4. Score vs Quality correlation
        if 'match_quality' in matches_df.columns:
            fig.add_trace(
                go.Box(
                    x=matches_df['match_quality'], 
                    y=matches_df['match_score'],
                    name='Score Distribution by Quality'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800, 
            showlegend=True, 
            title_text="Match Quality Analysis Dashboard"
        )
        
        return fig
    
    def create_data_quality_dashboard(self, spr_quality: Dict[str, Any], cad_quality: Dict[str, Any]) -> go.Figure:
        """Create data quality comparison dashboard"""
        
        # Extract completeness metrics using actual column names
        metrics = ['street_name_completeness', 'house_completeness', 'building_completeness', 'avg_completeness']
        
        # Safely extract values with defaults
        spr_values = []
        cad_values = []
        
        for metric in metrics:
            spr_val = spr_quality.get('completeness_metrics', {}).get(metric, 0)
            cad_val = cad_quality.get('completeness_metrics', {}).get(metric, 0)
            spr_values.append(spr_val)
            cad_values.append(cad_val)
        
        # Convert to percentages and close the radar chart
        spr_values_pct = [v * 100 for v in spr_values] + [spr_values[0] * 100]
        cad_values_pct = [v * 100 for v in cad_values] + [cad_values[0] * 100]
        metrics_labels = metrics + [metrics[0]]  # Close the radar
        
        fig = go.Figure()
        
        # SPR Registry trace
        fig.add_trace(go.Scatterpolar(
            r=spr_values_pct,
            theta=metrics_labels,
            fill='toself',
            name='SPR Registry',
            line_color='blue',
            marker_color='rgba(0, 0, 255, 0.3)'
        ))
        
        # Cadastre Registry trace
        fig.add_trace(go.Scatterpolar(
            r=cad_values_pct,
            theta=metrics_labels,
            fill='toself',
            name='Cadastre Registry',
            line_color='red',
            marker_color='rgba(255, 0, 0, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            showlegend=True,
            title="Data Quality Comparison Dashboard"
        )
        
        return fig
    
    def create_matching_statistics_chart(self, stats: Dict[str, Any]) -> go.Figure:
        """Create matching statistics visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Record Counts', 
                'Index Sizes',
                'Processing Overview',
                'Match Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # 1. Record counts
        record_types = ['SPR Total', 'SPR Processed', 'CAD Total']
        record_counts = [
            stats.get('total_spr_records', 0),
            stats.get('processed_spr_records', 0),
            stats.get('total_cad_records', 0)
        ]
        
        fig.add_trace(
            go.Bar(
                x=record_types,
                y=record_counts,
                name='Record Counts',
                marker_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # 2. Index sizes
        index_types = ['Street Index', 'House Index', 'Search Key Index', 'Component Index']
        index_sizes = [
            stats.get('street_index_size', 0),
            stats.get('house_index_size', 0),
            stats.get('search_key_index_size', 0),
            stats.get('component_index_size', 0)
        ]
        
        fig.add_trace(
            go.Bar(
                x=index_types,
                y=index_sizes,
                name='Index Sizes',
                marker_color=self.color_palette['secondary']
            ),
            row=1, col=2
        )
        
        # 3. Processing overview pie chart
        processing_data = {
            'Processed': stats.get('processed_spr_records', 0),
            'Remaining': stats.get('total_spr_records', 0) - stats.get('processed_spr_records', 0)
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(processing_data.keys()),
                values=list(processing_data.values()),
                name='Processing Status'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Matching Statistics Dashboard"
        )
        
        return fig
    
    def create_address_completeness_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create address completeness heatmap using actual column names"""
        if df.empty:
            return go.Figure()
        
        # Define columns to analyze (using actual column names from original system)
        address_columns = ['STREET_NAME', 'HOUSE', 'BUILDING']
        available_columns = [col for col in address_columns if col in df.columns]
        
        if not available_columns:
            return go.Figure()
        
        # Calculate completeness for each column
        completeness_data = []
        for col in available_columns:
            completeness = df[col].notna().sum() / len(df)
            completeness_data.append(completeness)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[completeness_data],
            x=available_columns,
            y=['Completeness'],
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=[[f'{val:.1%}' for val in completeness_data]],
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Completeness", tickformat='.0%')
        ))
        
        fig.update_layout(
            title="Address Field Completeness Analysis",
            xaxis_title="Address Fields",
            yaxis_title=""
        )
        
        return fig
    
    def create_duplicate_analysis_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create duplicate analysis visualization"""
        if df.empty or 'FULL_ADDRESS' not in df.columns:
            return go.Figure()
        
        # Analyze duplicates
        address_counts = df['FULL_ADDRESS'].value_counts()
        duplicate_counts = address_counts[address_counts > 1]
        
        if duplicate_counts.empty:
            # No duplicates found
            fig = go.Figure()
            fig.add_annotation(
                text="No duplicate addresses found",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Duplicate Address Analysis")
            return fig
        
        # Create histogram of duplicate frequencies
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=duplicate_counts.values,
            nbinsx=min(20, len(duplicate_counts)),
            name='Duplicate Frequency',
            marker_color=self.color_palette['poor']
        ))
        
        fig.update_layout(
            title=f"Duplicate Address Analysis ({len(duplicate_counts)} unique addresses have duplicates)",
            xaxis_title="Number of Duplicates",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def create_match_score_distribution(self, matches_df: pd.DataFrame) -> go.Figure:
        """Create detailed match score distribution"""
        if matches_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Score Distribution', 'Cumulative Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=matches_df['match_score'],
                nbinsx=30,
                name='Score Distribution',
                marker_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # Cumulative distribution
        sorted_scores = np.sort(matches_df['match_score'])
        y_values = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_scores,
                y=y_values,
                mode='lines',
                name='Cumulative Distribution',
                line=dict(color=self.color_palette['secondary'])
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            title_text="Match Score Distribution Analysis"
        )
        
        return fig
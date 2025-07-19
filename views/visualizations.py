import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np







def create_street_analysis_chart(matches_df):
    """Create street-level analysis visualization"""
    if len(matches_df) == 0:
        return None

    # Analyze most frequently matched streets
    street_counts = matches_df['STREET_NAME_SPR'].value_counts().head(20)
    
    fig = px.bar(
        x=street_counts.values,
        y=street_counts.index,
        orientation='h',
        title='Top 20 Most Frequently Matched Streets',
        labels={'x': 'Number of Matches', 'y': 'Street Name'}
    )

    fig.update_layout(
        xaxis_title="Number of Matches",
        yaxis_title="Street Name",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_quality_metrics_dashboard(spr_quality, cad_quality):
    """Create comprehensive quality metrics dashboard"""
    
    # Create subplots for multiple metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Record Completeness Comparison',
            'Unique Streets Comparison',
            'Data Quality Metrics',
            'Duplicate Records Analysis'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "radar"}, {"type": "pie"}]]
    )

    # Completeness comparison
    fig.add_trace(
        go.Bar(
            x=['SPR', 'Cadastre'],
            y=[spr_quality['avg_completeness'], cad_quality['avg_completeness']],
            name='Average Completeness',
            marker_color=['blue', 'red']
        ),
        row=1, col=1
    )

    # Unique streets comparison
    fig.add_trace(
        go.Bar(
            x=['SPR', 'Cadastre'],
            y=[spr_quality['unique_streets'], cad_quality['unique_streets']],
            name='Unique Streets',
            marker_color=['lightblue', 'lightcoral']
        ),
        row=1, col=2
    )

    # Quality metrics radar
    metrics = ['street_completeness', 'house_completeness', 'building_completeness']
    spr_values = [spr_quality[m] * 100 for m in metrics]
    cad_values = [cad_quality[m] * 100 for m in metrics]

    fig.add_trace(
        go.Scatterpolar(
            r=spr_values + [spr_values[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name='SPR Registry',
            line_color='blue'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatterpolar(
            r=cad_values + [cad_values[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name='Cadastre Registry',
            line_color='red'
        ),
        row=2, col=1
    )

    # Duplicate records analysis
    duplicate_data = [
        spr_quality['duplicate_addresses'],
        cad_quality['duplicate_addresses']
    ]
    fig.add_trace(
        go.Pie(
            labels=['SPR Duplicates', 'Cadastre Duplicates'],
            values=duplicate_data,
            name='Duplicates'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Data Quality Analysis Dashboard"
    )

    return fig


def create_advanced_matching_analysis(matches_df):
    """Create advanced matching analysis visualization"""
    if len(matches_df) == 0:
        return None

    # Create comprehensive analysis with multiple subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Score vs Match Type',
            'Completeness Impact on Scores',
            'Candidates Evaluated Distribution',
            'Score Distribution by Match Type',
            'Match Success Rate by Hour',
            'Quality Score Correlation'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "violin"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # Score vs Match Type
    for i, match_type in enumerate(matches_df['MATCH_TYPE'].unique()):
        subset = matches_df[matches_df['MATCH_TYPE'] == match_type]
        fig.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset['MATCH_SCORE'],
                mode='markers',
                name=match_type,
                marker=dict(size=8)
            ),
            row=1, col=1
        )

    # Completeness Impact on Scores
    fig.add_trace(
        go.Scatter(
            x=matches_df['COMPLETENESS_SPR'] + matches_df['COMPLETENESS_CAD'],
            y=matches_df['MATCH_SCORE'],
            mode='markers',
            name='Score vs Total Completeness',
            marker=dict(
                size=8,
                color=matches_df['MATCH_SCORE'],
                colorscale='viridis',
                showscale=True
            )
        ),
        row=1, col=2
    )

    # Candidates Evaluated Distribution
    if 'CANDIDATES_COUNT' in matches_df.columns:
        fig.add_trace(
            go.Histogram(
                x=matches_df['CANDIDATES_COUNT'],
                name='Candidates Distribution',
                nbinsx=20
            ),
            row=2, col=1
        )

    # Score Distribution by Match Type
    for match_type in matches_df['MATCH_TYPE'].unique():
        subset = matches_df[matches_df['MATCH_TYPE'] == match_type]
        fig.add_trace(
            go.Violin(
                y=subset['MATCH_SCORE'],
                name=match_type,
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=2
        )

    # Match Success Rate by Hour (if timestamp available)
    if 'MATCH_TIMESTAMP' in matches_df.columns:
        matches_df['MATCH_HOUR'] = pd.to_datetime(matches_df['MATCH_TIMESTAMP'], format='ISO8601').dt.hour
        hourly_counts = matches_df['MATCH_HOUR'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                name='Matches by Hour'
            ),
            row=3, col=1
        )

    # Quality Score Correlation
    fig.add_trace(
        go.Scatter(
            x=matches_df['COMPLETENESS_SPR'],
            y=matches_df['MATCH_SCORE'],
            mode='markers',
            name='SPR Completeness vs Score',
            marker=dict(color='blue', size=6)
        ),
        row=3, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=matches_df['COMPLETENESS_CAD'],
            y=matches_df['MATCH_SCORE'],
            mode='markers',
            name='CAD Completeness vs Score',
            marker=dict(color='red', size=6)
        ),
        row=3, col=2
    )

    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Advanced Matching Analysis Dashboard"
    )

    return fig
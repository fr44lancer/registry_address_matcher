import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_match_quality_chart(matches_df):
    """Create match quality visualization"""
    if len(matches_df) == 0:
        return None

    # Score distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Match Score Distribution', 'Match Type Distribution',
                        'Completeness Analysis', 'Matches Over Time'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Score distribution histogram
    fig.add_trace(
        go.Histogram(x=matches_df['MATCH_SCORE'], nbinsx=20, name='Score Distribution'),
        row=1, col=1
    )

    # Match type pie chart
    match_type_counts = matches_df['MATCH_TYPE'].value_counts()
    fig.add_trace(
        go.Pie(labels=match_type_counts.index, values=match_type_counts.values, name='Match Types'),
        row=1, col=2
    )

    # Completeness analysis
    fig.add_trace(
        go.Scatter(x=matches_df['COMPLETENESS_SPR'], y=matches_df['COMPLETENESS_CAD'],
                   mode='markers', name='Completeness Correlation'),
        row=2, col=1
    )

    # Matches over time (if timestamp available)
    if 'MATCH_TIMESTAMP' in matches_df.columns:
        matches_df['MATCH_HOUR'] = pd.to_datetime(matches_df['MATCH_TIMESTAMP']).dt.hour
        hourly_counts = matches_df['MATCH_HOUR'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='Hourly Matches'),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=True, title_text="Match Quality Analysis Dashboard")
    return fig


def create_data_quality_dashboard(spr_quality, cad_quality):
    """Create data quality comparison dashboard"""

    metrics = ['street_completeness', 'house_completeness', 'building_completeness', 'avg_completeness']
    spr_values = [spr_quality[m] for m in metrics]
    cad_values = [cad_quality[m] for m in metrics]

    # Convert to percentages and add the first value at the end to close the radar
    spr_values_pct = [v * 100 for v in spr_values] + [spr_values[0] * 100]
    cad_values_pct = [v * 100 for v in cad_values] + [cad_values[0] * 100]
    metrics_labels = metrics + [metrics[0]]  # Close the radar

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=spr_values_pct,
        theta=metrics_labels,
        fill='toself',
        name='SPR Registry',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=cad_values_pct,
        theta=metrics_labels,
        fill='toself',
        name='Cadastre Registry',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Data Quality Comparison (%)"
    )

    return fig
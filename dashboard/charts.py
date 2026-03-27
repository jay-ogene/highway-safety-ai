import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional


COLORS = {
    "critical": "#E24B4A",
    "high":     "#EF9F27",
    "warning":  "#378ADD",
    "safe":     "#639922",
    "teal":     "#1D9E75",
    "bg":       "rgba(0,0,0,0)"
}


def incidents_by_type_chart(summary: dict) -> go.Figure:
    """Donut chart of incidents by behavior type."""
    labels = [
        "Near-Miss", "Tailgating",
        "Sudden Braking", "Wrong Way", "Stopped Vehicle"
    ]
    values = [
        summary.get("near_miss_count", 0),
        summary.get("tailgating_count", 0),
        summary.get("sudden_braking_count", 0),
        summary.get("wrong_way_count", 0),
        summary.get("stopped_vehicle_count", 0),
    ]
    colors = ["#E24B4A","#EF9F27","#378ADD","#7F77DD","#1D9E75"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        height=280
    )
    return fig


def severity_bar_chart(summary: dict) -> go.Figure:
    """Horizontal bar chart of incidents by severity."""
    levels = ["Critical", "High", "Warning"]
    counts = [
        summary.get("critical_count", 0),
        summary.get("high_count", 0),
        summary.get("warning_count", 0),
    ]
    bar_colors = [
        COLORS["critical"],
        COLORS["high"],
        COLORS["warning"]
    ]

    fig = go.Figure(go.Bar(
        x=counts,
        y=levels,
        orientation="h",
        marker=dict(color=bar_colors),
        text=counts,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(t=10, b=10, l=10, r=40),
        height=200,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        font=dict(size=13)
    )
    return fig


def ttc_histogram(incidents_df: pd.DataFrame) -> go.Figure:
    """Histogram of Time-To-Collision values."""
    nm = incidents_df[
        (incidents_df["event_type"] == "near_miss") &
        (incidents_df["ttc_seconds"] > 0)
    ]["ttc_seconds"]

    if nm.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No near-miss data",
            showarrow=False, font=dict(size=14)
        )
        return fig

    fig = go.Figure(go.Histogram(
        x=nm,
        nbinsx=20,
        marker=dict(
            color=[
                COLORS["critical"] if v < 1.5
                else COLORS["high"] if v < 2.5
                else COLORS["warning"] if v < 3.0
                else COLORS["safe"]
                for v in nm
            ]
        ),
        hovertemplate="TTC: %{x:.2f}s<br>Count: %{y}<extra></extra>"
    ))

    # Add threshold lines
    for ttc, label, color in [
        (1.5, "CRITICAL", COLORS["critical"]),
        (2.5, "HIGH",     COLORS["high"]),
        (3.0, "WARNING",  COLORS["warning"])
    ]:
        fig.add_vline(
            x=ttc, line_dash="dash",
            line_color=color, line_width=1.5,
            annotation_text=label,
            annotation_font=dict(size=10, color=color)
        )

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(t=20, b=30, l=40, r=20),
        height=240,
        xaxis=dict(title="TTC (seconds)", showgrid=False),
        yaxis=dict(title="Count", showgrid=True,
                   gridcolor="rgba(128,128,128,0.2)"),
        bargap=0.05
    )
    return fig


def incidents_over_time(incidents_df: pd.DataFrame) -> go.Figure:
    """Line chart of incidents over frames."""
    if incidents_df.empty:
        return go.Figure()

    df = incidents_df.copy()
    df["frame_bucket"] = (df["frame_idx"] // 10) * 10
    grouped = df.groupby(
        ["frame_bucket", "event_type"]
    ).size().reset_index(name="count")

    fig = go.Figure()
    for etype, color in [
        ("near_miss", COLORS["critical"]),
        ("behavior",  COLORS["teal"])
    ]:
        sub = grouped[grouped["event_type"] == etype]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["frame_bucket"],
                y=sub["count"],
                mode="lines+markers",
                name=etype.replace("_", " ").title(),
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=(
                    "Frame: %{x}<br>"
                    "Count: %{y}<extra></extra>"
                )
            ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(t=10, b=30, l=40, r=20),
        height=220,
        xaxis=dict(
            title="Frame", showgrid=False
        ),
        yaxis=dict(
            title="Incidents",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)"
        ),
        legend=dict(
            orientation="h", y=1.1,
            font=dict(size=11)
        ),
        hovermode="x unified"
    )
    return fig


def speed_distribution(incidents_df: pd.DataFrame) -> go.Figure:
    """Histogram of closing speeds for near-miss events."""
    nm = incidents_df[
        (incidents_df["event_type"] == "near_miss") &
        (incidents_df["closing_speed_kmh"] > 0)
    ]["closing_speed_kmh"]

    if nm.empty:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=nm,
        nbinsx=15,
        marker=dict(color=COLORS["teal"]),
        hovertemplate=(
            "Closing speed: %{x:.0f} km/h"
            "<br>Count: %{y}<extra></extra>"
        )
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(t=10, b=30, l=40, r=20),
        height=200,
        xaxis=dict(title="Closing speed (km/h)", showgrid=False),
        yaxis=dict(title="Count", showgrid=True,
                   gridcolor="rgba(128,128,128,0.2)"),
        bargap=0.05
    )
    return fig
import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    return Path, go, json, mo, pl, px


@app.cell
def _(Path):
    results_dir = Path("../experiments/results/output")
    topic_keywords_path = results_dir / "topic_keywords.json"
    timeseries_path = results_dir / "topic_timeseries_H.csv"
    return timeseries_path, topic_keywords_path


@app.cell
def _(json, topic_keywords_path):
    with open(topic_keywords_path) as f:
        topic_keywords = {int(k): v for k, v in json.load(f).items()}
    return (topic_keywords,)


@app.cell
def _(pl, timeseries_path):
    ts_df = pl.read_csv(timeseries_path)
    return (ts_df,)


@app.cell
def _(mo):
    mo.md("""
    # Topic Modeling Results Explorer
    """)
    return


@app.cell
def _(mo, topic_keywords):
    mo.md(f"""
    ## Dataset Overview\n\n**Total Topics Discovered:** {len(topic_keywords)}
    """)
    return


@app.cell
def _(mo, pl, topic_keywords, ts_df):
    time_cols = [col for col in ts_df.columns if col != "topic"]
    topic_totals = ts_df.select([
        pl.col("topic"),
        pl.sum_horizontal(time_cols).alias("total_docs")
    ]).sort("total_docs", descending=True)

    topic_table_data = []
    for row in topic_totals.iter_rows(named=True):
        topic_id = row["topic"]
        keywords = ", ".join(topic_keywords.get(topic_id, [])[:5])
        topic_table_data.append({
            "Topic": topic_id,
            "Keywords": keywords,
            "Total Documents": row["total_docs"]
        })

    mo.ui.table(
        topic_table_data[:20],
        label="Top 20 Topics by Document Count",
        selection=None
    )
    return time_cols, topic_totals


@app.cell
def _(mo):
    topic_selector = mo.ui.slider(0, 99, value=0, label="Select Topic ID:", step=1)
    topic_selector
    return (topic_selector,)


@app.cell
def _(mo, topic_keywords, topic_selector):
    selected_topic = topic_selector.value
    keywords = topic_keywords.get(selected_topic, [])
    mo.md(f"### Topic {selected_topic} Keywords\n\n{', '.join(keywords)}")
    return (selected_topic,)


@app.cell
def _(go, pl, selected_topic, time_cols, ts_df):
    topic_data = ts_df.filter(pl.col("topic") == selected_topic)

    if topic_data.shape[0] > 0:
        values = [topic_data[col][0] if col in topic_data.columns else 0 for col in time_cols]

        fig_single = go.Figure()
        fig_single.add_trace(go.Scatter(
            x=time_cols,
            y=values,
            mode='lines+markers',
            name=f"Topic {selected_topic}",
            line=dict(color='#636EFA', width=2),
            marker=dict(size=4)
        ))

        fig_single.update_layout(
            title=f"Topic {selected_topic} Activity Over Time",
            xaxis_title="Time",
            yaxis_title="Document Count",
            template='plotly_white',
            height=400,
            hovermode='x'
        )

        fig_single
    return


@app.cell
def _(mo):
    mo.md("""
    ## Topic Distribution Analysis
    """)
    return


@app.cell
def _(px, topic_totals):
    fig_dist = px.bar(
        topic_totals.to_pandas(),
        x="topic",
        y="total_docs",
        title="Document Distribution Across Topics",
        labels={"topic": "Topic ID", "total_docs": "Total Documents"},
        template="plotly_white",
        height=500
    )
    fig_dist.update_traces(marker_color='#636EFA')
    fig_dist
    return


@app.cell
def _(mo):
    mo.md("""
    ## Temporal Dynamics - All Topics
    """)
    return


@app.cell
def _(mo):
    top_k_slider = mo.ui.slider(5, 20, value=10, label="Number of top topics to display:", step=1)
    top_k_slider
    return (top_k_slider,)


@app.cell
def _(go, time_cols, top_k_slider, topic_keywords, topic_totals, ts_df):
    top_k = top_k_slider.value
    top_topics_list = topic_totals.head(top_k)["topic"].to_list()

    fig_multi = go.Figure()

    for topic in top_topics_list:
        topic_row = ts_df.filter(ts_df["topic"] == topic)
        if topic_row.shape[0] > 0:
            vals = [topic_row[col][0] if col in topic_row.columns else 0 for col in time_cols]
            kws = topic_keywords.get(topic, [])[:3]
            label = f"Topic {topic}: {', '.join(kws)}"

            fig_multi.add_trace(go.Scatter(
                x=time_cols,
                y=vals,
                mode='lines',
                name=label,
                hovertemplate=f'{label}<br>Time: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))

    fig_multi.update_layout(
        title=f"Top {top_k} Topics Over Time",
        xaxis_title="Time",
        yaxis_title="Document Count",
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
    )

    fig_multi
    return


if __name__ == "__main__":
    app.run()

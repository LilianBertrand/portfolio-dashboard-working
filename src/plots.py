import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def performance_chart(returns_dict):
    df = pd.DataFrame({k: (1 + v).cumprod() * 100 for k, v in returns_dict.items()})
    fig = px.line(df, labels={"value": "Index level", "index": "Date", "variable": "Strategy"}, title="Portfolio Performance Base 100")
    return fig

def weights_bar(weights, title="Portfolio weights"):
    fig = px.bar(weights.sort_values(ascending=False), labels={"value": "Weight", "index": "Asset"}, title=title)
    fig.update_layout(showlegend=False)
    return fig

def corr_heatmap(returns):
    fig = px.imshow(returns.corr(), text_auto=".2f", aspect="auto", title="Correlation Matrix")
    return fig

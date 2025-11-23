import numpy as np
import plotly.graph_objects as go

def plot_wireframe(xs_2d, ys_2d, zs_2d, fig=None, linecolor="rgba(255,255,255,0.2)"):
    if fig is None:
        fig = go.Figure()

    for idx in range(np.shape(zs_2d)[0]):
        fig.add_trace(go.Scatter3d(x=xs_2d[idx, :], y=ys_2d[idx, :], z=zs_2d[idx, :], opacity=0.5,
                                   mode="lines", line=dict(color=linecolor, width=1), name=""))

    for idx in range(np.shape(zs_2d)[1]):
        fig.add_trace(go.Scatter3d(x=xs_2d[:,idx], y=ys_2d[:,idx], z=zs_2d[:,idx], opacity=0.5,
                                   mode="lines", line=dict(color=linecolor, width=1), name=""))

    return fig


def plot_surface(xs_2d, ys_2d, zs_2d, fig=None, colorscale="Viridis", opacity=0.8):
    if fig is None:
        fig = go.Figure()

    fig.add_trace(go.Surface(x=xs_2d, y=ys_2d, z=zs_2d, surfacecolor= (xs_2d**2+ys_2d**2)**0.5,colorscale=colorscale, opacity=opacity, showscale=False))

    return fig
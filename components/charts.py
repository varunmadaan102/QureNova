import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from components.theme import style_figure


def _ellipsoid_mesh(center, radii, color, name, opacity=0.55, resolution=16):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return go.Mesh3d(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo="name",
        alphahull=0,
    )


def conceptual_anatomy_figure():
    """Return a stylized, non-clinical body orientation visualization."""
    figure = go.Figure()
    figure.add_trace(_ellipsoid_mesh((0, 0, 1.3), (1.0, 0.42, 1.5), "#167C80", "Conceptual torso"))
    figure.add_trace(_ellipsoid_mesh((0, 0, 3.15), (0.48, 0.4, 0.5), "#29B7A8", "Conceptual head"))
    figure.add_trace(_ellipsoid_mesh((-0.64, -0.02, 2.0), (0.28, 0.32, 0.5), "#43D9DF", "Left chest region"))
    figure.add_trace(_ellipsoid_mesh((0.64, -0.02, 2.0), (0.28, 0.32, 0.5), "#43D9DF", "Right chest region"))
    for side in (-1, 1):
        figure.add_trace(
            go.Scatter3d(
                x=[side * 0.82, side * 1.22],
                y=[0, 0],
                z=[2.25, 0.65],
                mode="lines",
                line={"color": "#7A9AAF", "width": 12},
                name="Conceptual limb",
                showlegend=side == -1,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title="Conceptual anatomical orientation (not patient anatomy)",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
        },
        height=520,
        showlegend=True,
        legend={"orientation": "h", "y": -0.05},
    )
    return style_figure(figure)


def _sphere_mesh(center, radius, color, name, opacity=0.65, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return go.Mesh3d(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo="name",
        alphahull=0,
    )


def biomedical_visual_figure(view="Layered tissue", lesion_radius=0.22, opacity=0.58):
    """Build a generated educational tissue/tumor visualization.

    No patient images or clinical segmentation are used. The geometry is
    procedural and intentionally labeled as a visual simulation.
    """
    figure = go.Figure()
    if view == "Synthetic tumor model":
        figure.add_trace(
            _ellipsoid_mesh(
                (0, 0, 0),
                (1.35, 0.78, 0.9),
                "#D6789A",
                "Synthetic tissue volume",
                resolution=22,
            )
        )
        figure.add_trace(
            _sphere_mesh(
                (0.35, -0.45, 0.12),
                float(lesion_radius),
                "#FF5E6C",
                "Synthetic nodule (not a diagnosis)",
                opacity=0.95,
            )
        )
        figure.add_trace(
            _sphere_mesh(
                (-0.32, -0.55, -0.18),
                float(lesion_radius) * 0.55,
                "#FFB347",
                "Synthetic benign-looking variation",
                opacity=0.8,
            )
        )
    else:
        figure.add_trace(
            _ellipsoid_mesh(
                (0, 0, 0),
                (1.35, 0.78, 0.9),
                "#C96E9A",
                "Outer tissue layer",
                opacity=opacity,
                resolution=22,
            )
        )
        figure.add_trace(
            _ellipsoid_mesh(
                (0, -0.04, 0),
                (1.05, 0.62, 0.7),
                "#E7A2B9",
                "Inner tissue layer",
                opacity=min(0.9, opacity + 0.12),
                resolution=22,
            )
        )
        figure.add_trace(
            _sphere_mesh(
                (0, -0.52, 0),
                0.18,
                "#F6D38B",
                "Educational reference point",
                opacity=0.9,
            )
        )
    figure.update_layout(
        title=f"{view} · procedural educational simulation",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.0}},
        },
        height=540,
        legend={"orientation": "h", "y": -0.05},
    )
    return style_figure(figure)


def synthetic_xray_figure(lesion_radius=0.22):
    """Build a non-clinical grayscale projection for visual explanation."""
    axis = np.linspace(-2.0, 2.0, 180)
    x, y = np.meshgrid(axis, axis)
    tissue = np.exp(-((x / 1.25) ** 2 + (y / 1.55) ** 2) * 2.2)
    rib_pattern = 0.10 * (np.cos(x * 9.0) ** 2) * np.exp(-(y / 1.5) ** 2)
    nodule = 0.45 * np.exp(
        -(((x - 0.45) / max(0.08, lesion_radius)) ** 2
          + ((y + 0.25) / max(0.08, lesion_radius)) ** 2) * 2.0
    )
    image = np.clip(tissue + rib_pattern + nodule, 0, 1)
    figure = go.Figure(
        go.Heatmap(
            z=image,
            x=axis,
            y=axis,
            colorscale="Gray",
            showscale=False,
            hovertemplate="Synthetic projection<extra></extra>",
        )
    )
    figure.update_layout(
        title="Synthetic X-ray-style projection (not a medical image)",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        height=540,
    )
    return style_figure(figure)

def benchmark_bar(results):
    rows = []
    for name, result in results.items():
        for metric in ("accuracy", "f1", "roc_auc"):
            value = result.get("summary", {}).get(metric, {}).get("mean")
            if value is not None:
                rows.append({"Model": name, "Metric": metric.upper(), "Score": value})
    return style_figure(
        px.bar(pd.DataFrame(rows), x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1])
    )

def stability_chart(results):
    rows = []
    for name, result in results.items():
        metric = result.get("summary", {}).get("f1", {})
        if metric.get("mean") is not None:
            rows.append({"Model": name, "F1 Mean": metric["mean"], "F1 Std": metric["std"]})
    frame = pd.DataFrame(rows)
    return style_figure(px.bar(frame, x="Model", y="F1 Mean", error_y="F1 Std", range_y=[0, 1]))

def kernel_heatmap(matrix):
    return style_figure(
        px.imshow(
            matrix,
            aspect="auto",
            color_continuous_scale=["#07111f", "#0b283b", "#12445a", "#208f96", "#29b7a8", "#43d9df"],
            labels={"color": "Similarity"},
        )
    )


def feature_space_figure(
    frame,
    target_column=None,
    feature_columns=None,
    max_rows=300,
    pca_coords=None,
):
    """Build a bounded 3D orientation chart from the currently loaded data.

    The chart intentionally shows feature-space coordinates rather than anatomy,
    evidence, or a clinical prediction. PCA is fitted only for this view and
    does not change any experiment or prediction artifact.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("A non-empty DataFrame is required.")

    data = frame.head(int(max_rows)).copy()
    if feature_columns is None:
        feature_columns = [
            column
            for column in data.select_dtypes(include="number").columns
            if column != target_column
        ]
    feature_columns = [column for column in feature_columns if column in data.columns]
    if len(feature_columns) < 3:
        raise ValueError("At least three numeric features are required.")

    numeric = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 3:
        raise ValueError("At least three numeric features with usable values are required.")

    if pca_coords is not None:
        coordinates = pd.DataFrame(pca_coords).iloc[: len(data), :3].copy()
        coordinates.columns = ["PC1", "PC2", "PC3"][: coordinates.shape[1]]
        if coordinates.shape[1] < 3:
            pca_coords = None

    if pca_coords is None:
        transformed = SimpleImputer(strategy="median").fit_transform(numeric)
        transformed = StandardScaler().fit_transform(transformed)
        components = min(3, transformed.shape[0], transformed.shape[1])
        if components < 3:
            raise ValueError("Three PCA coordinates are not available for this dataset.")
        coordinates = pd.DataFrame(
            PCA(n_components=components, random_state=42).fit_transform(transformed),
            columns=["PC1", "PC2", "PC3"],
            index=data.index,
        )
        axis_labels = ["PC1", "PC2", "PC3"]
        source_label = "PCA coordinates (view-only)"
    else:
        axis_labels = list(coordinates.columns[:3])
        source_label = "PCA coordinates (provided)"

    plot_frame = coordinates.iloc[:, :3].copy()
    plot_frame["Row"] = [str(index) for index in data.index[: len(plot_frame)]]
    color_column = None
    if target_column and target_column in data.columns:
        plot_frame["Target"] = data[target_column].astype("string").fillna("Missing").tolist()[
            : len(plot_frame)
        ]
        color_column = "Target"

    figure = px.scatter_3d(
        plot_frame,
        x=axis_labels[0],
        y=axis_labels[1],
        z=axis_labels[2],
        color=color_column,
        hover_name="Row",
        title="Feature-space orientation (not anatomy or a clinical prediction)",
        labels={
            axis_labels[0]: axis_labels[0],
            axis_labels[1]: axis_labels[1],
            axis_labels[2]: axis_labels[2],
            "Target": "Target (dataset label)",
        },
    )
    figure.update_traces(marker=dict(size=5, opacity=0.78))
    figure.update_layout(
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        ),
        legend_title_text="Target (dataset label)" if color_column else None,
        title=f"{figure.layout.title.text} · {source_label}",
    )
    return style_figure(figure)

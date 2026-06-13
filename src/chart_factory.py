from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .risk_engine import percentile_of_value, zscore_of_value
from .seasonal_engine import remove_feb29, seasonal_matrix, seasonal_stats


SNAPSHOT_COLORS = {
    "最新": "#e31a1c",
    "三天前": "#e07a2f",
    "一周前": "#a9a9a9",
    "两周前": "#f2b701",
    "一个月前": "#4673c7",
    "两个月前": "#5fa54a",
    "三个月前": "#255f95",
}
HISTORY_COLORS = ["#8fa7c9", "#9fb9b4", "#c8b98d", "#b9adc9", "#a7a7a7"]
ASPECT_SIZES = {
    "PPT 16:9": (1280, 720),
    "Dashboard 宽屏": (1600, 900),
}


@dataclass
class ChartArtifact:
    chart_id: str
    title: str
    fig: go.Figure
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"
    message: str = ""
    source_series: str = ""


def _clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index, errors="coerce")
        clean = clean[clean.index.notna()]
    return clean.sort_index()


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}%"


def _delta(series: pd.Series, observations: int) -> float:
    clean = _clean_series(series)
    if len(clean) <= observations:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-observations - 1])


def _window_for_years(years: int) -> int:
    return int(252 * years)


def _metric_snapshot(label: str, series: pd.Series, reference_date: pd.Timestamp | None = None) -> dict[str, Any]:
    clean = _clean_series(series)
    if clean.empty:
        return {
            "指标": label,
            "最新值": "数据缺失",
            "1D变化": "N/A",
            "5D变化": "N/A",
            "20D变化": "N/A",
            "1Y分位": "N/A",
            "3Y分位": "N/A",
            "5Y分位": "N/A",
            "Z-score": "N/A",
            "季节性偏离": "N/A",
            "数据日期": "N/A",
            "数据状态": "数据缺失",
            "_raw": {},
        }

    latest_date = clean.index.max()
    latest_value = float(clean.iloc[-1])
    seasonal = seasonal_stats(clean, years=5)
    ref = pd.Timestamp(reference_date) if reference_date is not None else latest_date
    lag_days = int(max((ref.normalize() - latest_date.normalize()).days, 0))
    status = "正常" if lag_days <= 1 else f"数据滞后{lag_days}天"
    raw = {
        "latest": latest_value,
        "d1": _delta(clean, 1),
        "d5": _delta(clean, 5),
        "d20": _delta(clean, 20),
        "pct_1y": percentile_of_value(clean, latest_value, min(_window_for_years(1), len(clean))),
        "pct_3y": percentile_of_value(clean, latest_value, min(_window_for_years(3), len(clean))),
        "pct_5y": percentile_of_value(clean, latest_value, min(_window_for_years(5), len(clean))),
        "zscore": zscore_of_value(clean, latest_value, min(60, len(clean))),
        "seasonal_deviation": seasonal.get("seasonal_deviation", np.nan),
        "latest_date": latest_date.date().isoformat(),
        "status": status,
    }
    return {
        "指标": label,
        "最新值": _format_number(raw["latest"]),
        "1D变化": _format_number(raw["d1"]),
        "5D变化": _format_number(raw["d5"]),
        "20D变化": _format_number(raw["d20"]),
        "1Y分位": _format_percent(raw["pct_1y"]),
        "3Y分位": _format_percent(raw["pct_3y"]),
        "5Y分位": _format_percent(raw["pct_5y"]),
        "Z-score": _format_number(raw["zscore"]),
        "季节性偏离": _format_number(raw["seasonal_deviation"]),
        "数据日期": raw["latest_date"],
        "数据状态": status,
        "_raw": raw,
    }


def _apply_report_layout(fig: go.Figure, title: str, aspect: str = "PPT 16:9", showlegend: bool = True) -> go.Figure:
    width, height = ASPECT_SIZES.get(aspect, ASPECT_SIZES["PPT 16:9"])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=24, color="#3f3f3f")),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=64, r=36, t=74, b=92),
        font=dict(family="Microsoft YaHei, Arial, sans-serif", size=14, color="#555555"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        showlegend=showlegend,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(gridcolor="#d8d8d8", zeroline=False, tickfont=dict(size=13)),
    )
    return fig


def _placeholder_chart(chart_id: str, title: str, message: str, aspect: str = "PPT 16:9") -> ChartArtifact:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=22, color="#777777"))
    _apply_report_layout(fig, title, aspect=aspect, showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return ChartArtifact(chart_id=chart_id, title=title, fig=fig, status="missing", message=message)


def build_seasonal_chart(
    series: pd.Series,
    title: str,
    mode: str = "price",
    chart_id: str = "seasonal",
    aspect: str = "PPT 16:9",
    years: int = 5,
    show_mean: bool = True,
    show_iqr: bool = True,
    show_decile_band: bool = True,
) -> ChartArtifact:
    clean = _clean_series(series)
    if clean.empty:
        return _placeholder_chart(chart_id, title, "数据缺失/未找到序列", aspect)

    seasonal_source = remove_feb29(clean.to_frame("value"))["value"]
    matrix = seasonal_matrix(seasonal_source, years=years, interpolate=True)
    if matrix.empty:
        return _placeholder_chart(chart_id, title, "数据缺失/季节性样本不足", aspect)

    plot_index = pd.to_datetime("2001-" + matrix.index.astype(str))
    current_year = int(clean.index.max().year)
    fig = go.Figure()
    quantile_source = matrix.copy()

    if show_decile_band:
        q10 = quantile_source.quantile(0.10, axis=1)
        q90 = quantile_source.quantile(0.90, axis=1)
        fig.add_trace(go.Scatter(x=plot_index, y=q10, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=plot_index,
                y=q90,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(160, 160, 160, 0.12)",
                line=dict(width=0),
                name="10%-90%",
                hoverinfo="skip",
            )
        )
    if show_iqr:
        q25 = quantile_source.quantile(0.25, axis=1)
        q75 = quantile_source.quantile(0.75, axis=1)
        fig.add_trace(go.Scatter(x=plot_index, y=q25, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=plot_index,
                y=q75,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(225, 26, 28, 0.08)",
                line=dict(width=0),
                name="25%-75%",
                hoverinfo="skip",
            )
        )
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=plot_index,
                y=matrix.mean(axis=1),
                mode="lines",
                name="历史均值",
                line=dict(color="#444444", width=2, dash="dash"),
            )
        )

    history_index = 0
    for year in matrix.columns:
        year_int = int(year)
        values = matrix[year]
        if year_int == current_year:
            fig.add_trace(
                go.Scatter(
                    x=plot_index,
                    y=values,
                    mode="lines+markers",
                    name=str(year_int),
                    line=dict(color="#e31a1c", width=4),
                    marker=dict(size=5),
                    connectgaps=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_index,
                    y=values,
                    mode="lines",
                    name=str(year_int),
                    line=dict(color=HISTORY_COLORS[history_index % len(HISTORY_COLORS)], width=1.8),
                    opacity=0.42,
                    connectgaps=False,
                )
            )
            history_index += 1

    latest_date = clean.index.max()
    latest_key = latest_date.strftime("%m-%d")
    latest_x = pd.to_datetime(f"2001-{latest_key}", errors="coerce")
    if latest_key in matrix.index and pd.notna(latest_x):
        fig.add_trace(
            go.Scatter(
                x=[latest_x],
                y=[float(clean.iloc[-1])],
                mode="markers+text",
                text=["最新"],
                textposition="top center",
                name="最新点",
                marker=dict(color="#e31a1c", size=11),
                showlegend=False,
            )
        )

    _apply_report_layout(fig, title, aspect=aspect)
    fig.update_xaxes(tickformat="%b", dtick="M1", title_text="")
    fig.update_yaxes(title_text="value")
    meta = _metric_snapshot(title, clean)["_raw"]
    meta.update({"mode": mode, "source_series": getattr(series, "name", "") or title})
    return ChartArtifact(chart_id=chart_id, title=title, fig=fig, meta=meta, source_series=meta["source_series"])


def _nearest_snapshot_date(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    valid = pd.DatetimeIndex(index.dropna()).sort_values()
    if valid.empty:
        return None
    prior = valid[valid <= target]
    if not prior.empty:
        return prior[-1]
    return valid[0]


def _column_for_tenor(prefix: str, tenor: str, columns: pd.Index) -> str | None:
    candidates = [
        f"{prefix}{tenor}",
        f"{prefix}_{tenor}",
        f"{prefix}-{tenor}",
        f"{prefix}_{tenor}_spread",
        f"{prefix}{tenor}_spread",
    ]
    colset = {str(col): col for col in columns}
    for candidate in candidates:
        if candidate in colset:
            return colset[candidate]
    return None


def _snapshot_label(base_label: str, date: pd.Timestamp) -> str:
    return f"{date.year}/{date.month}/{date.day} {base_label}"


def build_curve_snapshot_chart(
    curve_family: str,
    tenors: list[str],
    snapshot_offsets: list[dict[str, Any]],
    frame: pd.DataFrame,
    title: str | None = None,
    chart_id: str = "curve_snapshot",
    aspect: str = "PPT 16:9",
    display_family: str | None = None,
) -> ChartArtifact:
    if frame is None or frame.empty:
        return _placeholder_chart(chart_id, title or curve_family, "数据缺失/未找到序列", aspect)
    data = frame.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")
        data = data[data.index.notna()]
    data = data.sort_index()
    latest_date = data.dropna(how="all").index.max() if not data.dropna(how="all").empty else None
    if latest_date is None or pd.isna(latest_date):
        return _placeholder_chart(chart_id, title or curve_family, "数据缺失/未找到序列", aspect)

    family_label = display_family or curve_family
    x_labels = [f"{family_label}{tenor}" for tenor in tenors]
    columns = [_column_for_tenor(curve_family, tenor, data.columns) for tenor in tenors]
    if not any(columns):
        return _placeholder_chart(chart_id, title or f"{family_label}期限结构", "数据缺失/未找到序列", aspect)

    fig = go.Figure()
    manifest_dates: list[str] = []
    for offset in snapshot_offsets:
        label = str(offset.get("label", ""))
        days = int(offset.get("days", 0))
        target = pd.Timestamp(latest_date) - pd.Timedelta(days=days)
        snap_date = _nearest_snapshot_date(data.index, target)
        if snap_date is None:
            continue
        row = data.loc[snap_date]
        y_values = [float(row[col]) if col is not None and pd.notna(row.get(col, np.nan)) else None for col in columns]
        if all(value is None for value in y_values):
            continue
        is_latest = days == 0
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_values,
                mode="lines+markers",
                name=_snapshot_label(label, snap_date),
                line=dict(color=SNAPSHOT_COLORS.get(label, "#777777"), width=4 if is_latest else 3),
                marker=dict(size=8 if is_latest else 7),
                connectgaps=False,
            )
        )
        manifest_dates.append(snap_date.date().isoformat())

    if not fig.data:
        return _placeholder_chart(chart_id, title or f"{family_label}期限结构", "数据缺失/未找到序列", aspect)

    chart_title = title or f"{family_label}期限结构"
    _apply_report_layout(fig, chart_title, aspect=aspect)
    fig.update_xaxes(title_text="", tickangle=0)
    fig.update_yaxes(title_text="value")
    return ChartArtifact(
        chart_id=chart_id,
        title=chart_title,
        fig=fig,
        meta={"latest_date": pd.Timestamp(latest_date).date().isoformat(), "snapshot_dates": manifest_dates},
        source_series=", ".join([str(col) for col in columns if col is not None]),
    )


def build_spread_structure_chart(
    spread_family: str,
    tenors: list[str],
    snapshot_offsets: list[dict[str, Any]],
    frame: pd.DataFrame,
    title: str | None = None,
    chart_id: str = "spread_structure",
    aspect: str = "PPT 16:9",
    display_family: str | None = None,
) -> ChartArtifact:
    artifact = build_curve_snapshot_chart(
        curve_family=spread_family,
        tenors=tenors,
        snapshot_offsets=snapshot_offsets,
        frame=frame,
        title=title or f"{display_family or spread_family}价差结构",
        chart_id=chart_id,
        aspect=aspect,
        display_family=display_family or spread_family.replace("_", "-"),
    )
    artifact.fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="#888888")
    return artifact


def build_state_matrix(metrics: Mapping[str, pd.Series], reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = [_metric_snapshot(label, series, reference_date=reference_date) for label, series in metrics.items()]
    frame = pd.DataFrame(rows)
    return frame.drop(columns=["_raw"], errors="ignore")


def metric_raw_snapshots(metrics: Mapping[str, pd.Series], reference_date: pd.Timestamp | None = None) -> dict[str, dict[str, Any]]:
    return {label: _metric_snapshot(label, series, reference_date=reference_date)["_raw"] for label, series in metrics.items()}


def build_move_leaderboard(metrics: Mapping[str, pd.Series], window: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, series in metrics.items():
        clean = _clean_series(series)
        if clean.empty or len(clean) <= window:
            continue
        move = float(clean.iloc[-1] - clean.iloc[-window - 1])
        rows.append(
            {
                "指标": label,
                f"{window}D变化": move,
                "最新值": float(clean.iloc[-1]),
                "数据日期": clean.index[-1].date().isoformat(),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["绝对变化"] = frame[f"{window}D变化"].abs()
    return frame.sort_values("绝对变化", ascending=False).drop(columns=["绝对变化"]).head(12)


def build_move_leaderboard_chart(
    metrics: Mapping[str, pd.Series],
    window: int,
    chart_id: str,
    aspect: str = "PPT 16:9",
) -> ChartArtifact:
    table = build_move_leaderboard(metrics, window)
    title = f"{window}D变化排行榜"
    if table.empty:
        return _placeholder_chart(chart_id, title, "数据缺失/样本不足", aspect)
    display = table.copy()
    for column in [f"{window}D变化", "最新值"]:
        display[column] = display[column].map(_format_number)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(display.columns), fill_color="#f1f3f5", align="center", font=dict(size=15)),
                cells=dict(values=[display[col] for col in display.columns], align="center", height=30, font=dict(size=14)),
            )
        ]
    )
    _apply_report_layout(fig, title, aspect=aspect, showlegend=False)
    return ChartArtifact(chart_id=chart_id, title=title, fig=fig, meta={"window": window})


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return str(value)


def export_report_charts(
    page_type: str,
    output_dir: str | Path,
    chart_ids: list[str] | None = None,
    chart_results: list[ChartArtifact] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_ids = set(chart_ids or [])
    results = chart_results or []
    manifest: list[dict[str, Any]] = []
    exported: list[str] = []
    errors: list[str] = []

    for result in results:
        if selected_ids and result.chart_id not in selected_ids:
            continue
        filename = f"{page_type}_{result.chart_id}.png"
        file_path = output_path / filename
        try:
            result.fig.write_image(str(file_path), width=result.fig.layout.width, height=result.fig.layout.height, scale=2)
            exported.append(str(file_path))
        except Exception as exc:  # Plotly raises when kaleido is unavailable.
            errors.append(f"{result.chart_id}: {exc}")
            continue
        manifest.append(
            {
                "chart_id": result.chart_id,
                "title": result.title,
                "source_series": result.source_series,
                "latest_date": result.meta.get("latest_date"),
                "output_path": str(file_path),
                "status": result.status,
                "message": result.message,
            }
        )

    (output_path / "chart_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    summary = {"page_type": page_type, "output_dir": str(output_path), "exported": exported, "errors": errors}
    (output_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    for name, table in (tables or {}).items():
        csv_path = output_path / f"{name}.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if name == "freshness_report":
            try:
                table.to_excel(output_path / "freshness_report.xlsx", index=False)
            except Exception:
                pass
    return summary

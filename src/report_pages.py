from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .chart_factory import (
    ASPECT_SIZES,
    ChartArtifact,
    build_curve_snapshot_chart,
    build_move_leaderboard,
    build_move_leaderboard_chart,
    build_seasonal_chart,
    build_spread_structure_chart,
    build_state_matrix,
    export_report_charts,
    metric_raw_snapshots,
)


def _report_frames(
    sources: dict[str, pd.DataFrame],
    portfolios: pd.DataFrame,
    basis_formula_df: pd.DataFrame,
    merged_for_formula: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames = dict(sources)
    frames["portfolios"] = portfolios
    frames["basis_formula"] = basis_formula_df
    frames["merged"] = merged_for_formula
    return frames


def _resolve_series(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> tuple[str, pd.Series]:
    frame = frames.get(str(spec.get("source", "")))
    if frame is None or frame.empty:
        return "", pd.Series(dtype=float)
    for candidate in spec.get("candidates", []):
        if candidate in frame.columns:
            series = pd.to_numeric(frame[candidate], errors="coerce")
            series.name = str(candidate)
            return str(candidate), series
    return "", pd.Series(dtype=float)


def _metric_series_map(config: dict[str, Any], frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    metrics: dict[str, pd.Series] = {}
    for spec in config.get("state_metrics", []):
        _, series = _resolve_series(spec, frames)
        metrics[str(spec.get("label", spec.get("id", "")))] = series
    return metrics


def _reference_date(metrics: dict[str, pd.Series]) -> pd.Timestamp | None:
    latest_dates = []
    for series in metrics.values():
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if not clean.empty and isinstance(clean.index, pd.DatetimeIndex):
            latest_dates.append(clean.index.max())
    if not latest_dates:
        return None
    return pd.Timestamp(max(latest_dates))


def _freshness_report(config: dict[str, Any], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = _metric_series_map(config, frames)
    ref = _reference_date(metrics)
    rows: list[dict[str, Any]] = []
    for label, series in metrics.items():
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
            rows.append({"指标": label, "最新日期": "N/A", "滞后天数": "N/A", "数据状态": "数据缺失"})
            continue
        latest = pd.Timestamp(clean.index.max())
        lag = int(max(((ref or latest).normalize() - latest.normalize()).days, 0))
        status = "正常" if lag <= 1 else f"数据滞后{lag}天"
        rows.append({"指标": label, "最新日期": latest.date().isoformat(), "滞后天数": lag, "数据状态": status})
    return pd.DataFrame(rows)


def _objective_notes(raw_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for label, raw in raw_metrics.items():
        status = str(raw.get("status", ""))
        pct_1y = raw.get("pct_1y", np.nan)
        pct_3y = raw.get("pct_3y", np.nan)
        zscore = raw.get("zscore", np.nan)
        d5 = raw.get("d5", np.nan)
        if status.startswith("数据滞后") or status == "数据缺失":
            rows.append({"指标": label, "提示": status})
        if pd.notna(pct_3y) and (float(pct_3y) >= 80 or float(pct_3y) <= 20):
            rows.append({"指标": label, "提示": f"3Y {float(pct_3y):.1f}% 分位"})
        elif pd.notna(pct_1y) and (float(pct_1y) >= 80 or float(pct_1y) <= 20):
            rows.append({"指标": label, "提示": f"1Y {float(pct_1y):.1f}% 分位"})
        if pd.notna(zscore) and abs(float(zscore)) >= 2:
            rows.append({"指标": label, "提示": f"Z-score {float(zscore):.2f}"})
        if pd.notna(d5):
            rows.append({"指标": label, "提示": f"5D变化 {float(d5):.2f}"})
    if not rows:
        return pd.DataFrame([{"指标": "核心指标", "提示": "未触发高/低分位或数据滞后提示"}])
    return pd.DataFrame(rows).drop_duplicates().head(20)


def _meta_strip(artifact: ChartArtifact) -> None:
    meta = artifact.meta or {}
    items = [
        ("最新日期", meta.get("latest_date")),
        ("最新值", meta.get("latest")),
        ("5D变化", meta.get("d5")),
        ("20D变化", meta.get("d20")),
        ("1Y分位", meta.get("pct_1y")),
        ("3Y分位", meta.get("pct_3y")),
        ("季节性偏离", meta.get("seasonal_deviation")),
    ]
    rendered = []
    for label, value in items:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and "分位" in label:
            value_text = f"{float(value):.1f}%"
        elif isinstance(value, (int, float, np.integer, np.floating)):
            value_text = f"{float(value):,.2f}"
        else:
            value_text = str(value)
        rendered.append(f"<span><b>{label}</b> {value_text}</span>")
    if rendered:
        st.markdown(
            "<div class='report-meta-strip'>" + "".join(rendered) + "</div>",
            unsafe_allow_html=True,
        )


def _chart_from_spec(
    chart_id: str,
    spec: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    snapshot_offsets: list[dict[str, Any]],
    aspect: str,
) -> ChartArtifact:
    chart_type = str(spec.get("type", ""))
    title = str(spec.get("title", chart_id))
    source = str(spec.get("source", ""))
    frame = frames.get(source, pd.DataFrame())
    if chart_type == "seasonal":
        source_name, series = _resolve_series(spec, frames)
        artifact = build_seasonal_chart(series, title=title, chart_id=chart_id, aspect=aspect)
        artifact.source_series = source_name
        return artifact
    if chart_type == "curve_snapshot":
        return build_curve_snapshot_chart(
            curve_family=str(spec.get("curve_family", "")),
            display_family=spec.get("display_family"),
            tenors=[str(item) for item in spec.get("tenors", [])],
            tenor_mode=spec.get("tenor_mode"),
            tenor_count=int(spec.get("tenor_count", 12)),
            snapshot_offsets=snapshot_offsets,
            frame=frame,
            title=title,
            chart_id=chart_id,
            aspect=aspect,
        )
    if chart_type == "spread_structure":
        return build_spread_structure_chart(
            spread_family=str(spec.get("spread_family", "")),
            display_family=spec.get("display_family"),
            tenors=[str(item) for item in spec.get("tenors", [])],
            tenor_mode=spec.get("tenor_mode"),
            tenor_count=int(spec.get("tenor_count", 12)),
            snapshot_offsets=snapshot_offsets,
            frame=frame,
            title=title,
            chart_id=chart_id,
            aspect=aspect,
        )
    return build_seasonal_chart(pd.Series(dtype=float), title=title, chart_id=chart_id, aspect=aspect)


def _render_chart_card(page_type: str, artifact: ChartArtifact, output_dir: str, aspect: str) -> None:
    width, height = ASPECT_SIZES.get(aspect, ASPECT_SIZES["PPT 16:9"])
    st.plotly_chart(
        artifact.fig,
        use_container_width=True,
        key=f"report_chart_{page_type}_{artifact.chart_id}",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{page_type}_{artifact.chart_id}",
                "width": width,
                "height": height,
                "scale": 2,
            },
        },
    )
    _meta_strip(artifact)
    if artifact.status != "ready" and artifact.message:
        st.caption(artifact.message)
    if st.button("导出 PNG", key=f"export_{page_type}_{artifact.chart_id}"):
        result = export_report_charts(page_type, output_dir, [artifact.chart_id], [artifact])
        if result["errors"]:
            st.error("导出失败：" + "；".join(result["errors"]))
        else:
            st.success(f"已导出：{result['exported'][0]}")


def _render_controls(page_type: str, workbook_path: str, on_refresh) -> tuple[str, str, bool]:
    st.markdown(f"### {'周报出图' if page_type == 'weekly' else '日报出图'}")
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        if st.button("刷新 Excel 数据", key=f"{page_type}_refresh_excel"):
            ok = on_refresh()
            st.session_state[f"{page_type}_last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("Excel 已刷新" if ok else "Excel 刷新未完成")
            st.rerun()
    with col2:
        aspect = st.selectbox("图表比例", list(ASPECT_SIZES), key=f"{page_type}_aspect")
    with col3:
        default_dir = str(Path("outputs") / "report_charts" / page_type)
        output_dir = st.text_input("导出目录", value=default_dir, key=f"{page_type}_output_dir")
    with col4:
        export_all = st.button("导出本页所有图", key=f"{page_type}_export_all")
    last_refresh = st.session_state.get(f"{page_type}_last_refresh", "本次会话未刷新")
    st.caption(f"数据文件：{workbook_path} ｜ 最近一次刷新：{last_refresh}")
    return output_dir, aspect, export_all


def render_report_page(
    page_type: str,
    workbook_path: str,
    sources: dict[str, pd.DataFrame],
    portfolios: pd.DataFrame,
    basis_formula_df: pd.DataFrame,
    merged_for_formula: pd.DataFrame,
    report_config: dict[str, Any],
    on_refresh,
) -> None:
    frames = _report_frames(sources, portfolios, basis_formula_df, merged_for_formula)
    output_dir, aspect, export_all = _render_controls(page_type, workbook_path, on_refresh)

    metrics = _metric_series_map(report_config, frames)
    ref_date = _reference_date(metrics)
    state_matrix = build_state_matrix(metrics, reference_date=ref_date)
    freshness = _freshness_report(report_config, frames)
    raw = metric_raw_snapshots(metrics, reference_date=ref_date)
    leaderboards = {f"move_leaderboard_{window}d": build_move_leaderboard(metrics, window) for window in (1, 5, 20)}

    st.markdown("#### 顶部状态矩阵")
    st.dataframe(state_matrix, use_container_width=True, hide_index=True)

    if page_type == "daily":
        st.markdown("#### 数据新鲜度")
        st.dataframe(freshness, use_container_width=True, hide_index=True)
        st.markdown("#### 日度变化排行榜")
        cols = st.columns(3)
        for col, window in zip(cols, (1, 5, 20)):
            with col:
                table = leaderboards[f"move_leaderboard_{window}d"].copy()
                if table.empty:
                    st.info(f"{window}D样本不足")
                else:
                    st.caption(f"{window}D变化")
                    st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown("#### 客观提示")
        st.dataframe(_objective_notes(raw), use_container_width=True, hide_index=True)

    chart_results: list[ChartArtifact] = []
    if page_type == "daily":
        for window in (1, 5, 20):
            chart_results.append(build_move_leaderboard_chart(metrics, window, chart_id=f"move_leaderboard_{window}d", aspect=aspect))

    page_cfg = report_config.get(page_type, {})
    chart_specs = report_config.get("charts", {})
    snapshot_offsets = report_config.get("snapshot_offsets", [])
    for section in page_cfg.get("sections", []):
        st.markdown(f"#### {section.get('title', section.get('id', ''))}")
        for chart_id in section.get("charts", []):
            spec = chart_specs.get(chart_id)
            if spec is None:
                st.info(f"{chart_id}: 数据缺失/未找到配置")
                continue
            artifact = _chart_from_spec(chart_id, spec, frames, snapshot_offsets, aspect)
            chart_results.append(artifact)
            _render_chart_card(page_type, artifact, output_dir, aspect)

    tables = {
        "state_matrix": state_matrix,
        "freshness_report": freshness,
        **leaderboards,
    }
    if export_all:
        result = export_report_charts(page_type, output_dir, chart_results=chart_results, tables=tables)
        if result["errors"]:
            st.error("部分图表导出失败：" + "；".join(result["errors"]))
        st.success(f"已导出 {len(result['exported'])} 张图到 {output_dir}")

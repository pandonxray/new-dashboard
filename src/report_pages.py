from __future__ import annotations

import ctypes
import io
import math
import subprocess
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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


CF_DIB = 8
GMEM_MOVEABLE = 0x0002


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


def _normalize_tenor(value: Any) -> str:
    text = str(value).strip()
    return text.zfill(2) if text.isdigit() and len(text) <= 2 else text


def _relative_month_tenor(offset: int, reference_date: pd.Timestamp | None = None) -> str:
    anchor = pd.Timestamp(reference_date or pd.Timestamp.today())
    month = ((int(anchor.month) - 1 + int(offset)) % 12) + 1
    return f"{month:02d}"


def _resolve_dynamic_tenor(rule: dict[str, Any] | None) -> tuple[str, str]:
    rule = rule or {}
    offset = int(rule.get("offset", 0))
    selected_offset = offset
    tenor = _relative_month_tenor(offset)
    avoid = {_normalize_tenor(item) for item in rule.get("avoid", [])}
    if tenor in avoid:
        selected_offset = int(rule.get("fallback_offset", offset + 1))
        tenor = _relative_month_tenor(selected_offset)
    label = "M" if selected_offset == 0 else f"M+{selected_offset}"
    return tenor, label


def _format_title(template: str, **values: Any) -> str:
    try:
        return template.format(**values)
    except Exception:
        return template


def _find_tenor_column(prefix: str, tenor: str, columns: pd.Index) -> str | None:
    tenor = _normalize_tenor(tenor)
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
            return str(colset[candidate])
    return None


def _resolve_leg_series(leg: dict[str, Any], frames: dict[str, pd.DataFrame]) -> tuple[str, pd.Series, dict[str, str]]:
    frame = frames.get(str(leg.get("source", "")), pd.DataFrame())
    if frame is None or frame.empty:
        return "", pd.Series(dtype=float), {}
    if leg.get("family"):
        tenor = _normalize_tenor(leg.get("tenor", ""))
        month_label = ""
        if "tenor_offset" in leg:
            tenor = _relative_month_tenor(int(leg.get("tenor_offset", 0)))
            month_label = "M" if int(leg.get("tenor_offset", 0)) == 0 else f"M+{int(leg.get('tenor_offset', 0))}"
        column = _find_tenor_column(str(leg.get("family", "")), tenor, frame.columns)
        if column and column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            series.name = column
            return column, series, {"tenor": tenor, "month_label": month_label}
        return "", pd.Series(dtype=float), {"tenor": tenor, "month_label": month_label}
    for candidate in leg.get("candidates", []):
        if candidate in frame.columns:
            series = pd.to_numeric(frame[candidate], errors="coerce")
            series.name = str(candidate)
            return str(candidate), series, {}
    return "", pd.Series(dtype=float), {}


def _build_derived_spread(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> tuple[str, pd.Series, str]:
    left_name, left, left_meta = _resolve_leg_series(spec.get("left", {}), frames)
    right_name, right, right_meta = _resolve_leg_series(spec.get("right", {}), frames)
    if left.empty or right.empty:
        return "", pd.Series(dtype=float), ""
    left_aligned, right_aligned = left.align(right, join="outer")
    series = left_aligned - right_aligned
    default_name = f"{left_name}-{right_name}"
    series.name = str(spec.get("series_name") or default_name)
    title = _format_title(
        str(spec.get("title_template", spec.get("title", default_name))),
        left=left_name,
        right=right_name,
        left_tenor=left_meta.get("tenor", ""),
        right_tenor=right_meta.get("tenor", ""),
        left_month_label=left_meta.get("month_label", ""),
        right_month_label=right_meta.get("month_label", ""),
    )
    return title, series, default_name


def _prepare_seasonal_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prepared = dict(spec)
    title = str(prepared.get("title", "seasonal"))
    if prepared.get("dynamic_tenor"):
        tenor, month_label = _resolve_dynamic_tenor(prepared.get("dynamic_tenor"))
        if prepared.get("candidate_templates"):
            prepared["candidates"] = [str(template).format(tenor=tenor) for template in prepared.get("candidate_templates", [])]
        title = _format_title(str(prepared.get("title_template", title)), tenor=tenor, month_label=month_label)
    return prepared, title


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
        prepared, seasonal_title = _prepare_seasonal_spec(spec)
        source_name, series = _resolve_series(prepared, frames)
        artifact = build_seasonal_chart(series, title=seasonal_title, chart_id=chart_id, aspect=aspect)
        artifact.source_series = source_name
        return artifact
    if chart_type == "seasonal_spread":
        spread_title, series, source_name = _build_derived_spread(spec, frames)
        title = spread_title or title
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


def _figure_to_png_bytes(fig, aspect: str, scale: int = 2) -> bytes:
    width, height = ASPECT_SIZES.get(aspect, ASPECT_SIZES["PPT 16:9"])
    layout_width = int(fig.layout.width or width)
    layout_height = int(fig.layout.height or height)
    return fig.to_image(format="png", width=layout_width, height=layout_height, scale=scale)


def _copy_image_to_windows_clipboard(image: Image.Image) -> None:
    if sys.platform != "win32":
        raise RuntimeError("当前复制图片到剪贴板仅支持 Windows 本地运行。")

    # PowerPoint handles images placed through the Windows Forms clipboard more
    # reliably than a hand-written CF_DIB block in some Streamlit sessions.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        image.convert("RGB").save(tmp_path, "PNG")
        escaped = str(tmp_path).replace("'", "''")
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile('{escaped}')
try {{
    [System.Windows.Forms.Clipboard]::SetImage($img)
}} finally {{
    $img.Dispose()
}}
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return
    except Exception:
        pass
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    rgb = image.convert("RGB")
    output = io.BytesIO()
    rgb.save(output, "BMP")
    dib = output.getvalue()[14:]

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p

    opened = False
    for _ in range(12):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.06)
    if not opened:
        raise RuntimeError("剪贴板暂时被占用，请稍后再点一次复制。")

    handle = None
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib))
        if not handle:
            raise RuntimeError("申请剪贴板内存失败。")
        locked = kernel32.GlobalLock(handle)
        if not locked:
            kernel32.GlobalFree(handle)
            handle = None
            raise RuntimeError("锁定剪贴板内存失败。")
        ctypes.memmove(locked, dib, len(dib))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_DIB, handle):
            kernel32.GlobalFree(handle)
            handle = None
            raise RuntimeError("写入剪贴板失败。")
        handle = None
    finally:
        user32.CloseClipboard()


def _copy_artifact_to_clipboard(artifact: ChartArtifact, aspect: str) -> tuple[bool, str]:
    try:
        png_bytes = _figure_to_png_bytes(artifact.fig, aspect=aspect, scale=2)
        image = Image.open(io.BytesIO(png_bytes))
        _copy_image_to_windows_clipboard(image)
        return True, f"已复制：{artifact.title}"
    except Exception as exc:
        return False, f"复制失败：{exc}"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _truncate_text(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _paste_fit(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    max_width = max(right - left, 1)
    max_height = max(bottom - top, 1)
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    x = left + (max_width - fitted.width) // 2
    y = top + (max_height - fitted.height) // 2
    canvas.paste(fitted, (x, y))


def _build_section_contact_sheet(
    artifacts: list[ChartArtifact],
    section_title: str,
    description: str,
    aspect: str,
) -> Image.Image:
    base_width, base_height = ASPECT_SIZES.get(aspect, ASPECT_SIZES["PPT 16:9"])
    scale = 2
    width, height = base_width * scale, base_height * scale
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    note_font = _font(23)
    label_font = _font(22, bold=True)

    margin = 56
    header_h = 104
    gap = 26
    footer_h = 28
    count = max(len(artifacts), 1)
    columns = 2 if count in (2, 4) else 3 if count == 3 else min(count, 2)
    rows = math.ceil(count / columns)
    grid_w = width - margin * 2
    grid_h = height - header_h - footer_h - margin
    tile_w = int((grid_w - gap * (columns - 1)) / columns)
    tile_h = int((grid_h - gap * (rows - 1)) / rows)

    draw.text((margin, 34), section_title, fill="#243447", font=title_font)

    for idx, artifact in enumerate(artifacts):
        row, col = divmod(idx, columns)
        x0 = margin + col * (tile_w + gap)
        y0 = header_h + row * (tile_h + gap)
        x1 = x0 + tile_w
        y1 = y0 + tile_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#ffffff", outline="#e8e8e8", width=1)
        label = _truncate_text(artifact.title, 36)
        draw.text((x0 + 18, y0 + 14), label, fill="#243447", font=label_font)
        try:
            png_bytes = _figure_to_png_bytes(artifact.fig, aspect=aspect, scale=2)
            image = Image.open(io.BytesIO(png_bytes))
            _paste_fit(canvas, image, (x0 + 16, y0 + 56, x1 - 16, y1 - 14))
        except Exception as exc:
            draw.text((x0 + 24, y0 + 72), f"图表生成失败：{exc}", fill="#a05b4f", font=note_font)

    return canvas


def _copy_section_to_clipboard(artifacts: list[ChartArtifact], section: dict[str, Any], aspect: str) -> tuple[bool, str]:
    try:
        image = _build_section_contact_sheet(
            artifacts,
            str(section.get("title", section.get("id", "板块"))),
            str(section.get("description", "")),
            aspect,
        )
        _copy_image_to_windows_clipboard(image)
        return True, f"已复制板块：{section.get('title', section.get('id', ''))}"
    except Exception as exc:
        return False, f"复制板块失败：{exc}"


def _render_chart_card(
    page_type: str,
    artifact: ChartArtifact,
    output_dir: str,
    aspect: str,
    label: str | None = None,
    render_key: str | None = None,
) -> None:
    width, height = ASPECT_SIZES.get(aspect, ASPECT_SIZES["PPT 16:9"])
    element_key = render_key or f"{page_type}_{artifact.chart_id}"
    if label:
        st.caption(label)
    st.plotly_chart(
        artifact.fig,
        use_container_width=True,
        key=f"report_chart_{element_key}",
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
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("复制图片", key=f"copy_{element_key}"):
            ok, message = _copy_artifact_to_clipboard(artifact, aspect)
            if ok:
                st.success(message)
            else:
                st.error(message)
    with action_cols[1]:
        if st.button("导出 PNG", key=f"export_{element_key}"):
            result = export_report_charts(page_type, output_dir, [artifact.chart_id], [artifact])
            if result["errors"]:
                st.error("导出失败：" + "；".join(result["errors"]))
            else:
                st.success(f"已导出：{result['exported'][0]}")


def _section_grid_columns(count: int) -> int:
    if count == 3:
        return 3
    if count >= 2:
        return 2
    return 1


def _build_section_artifacts(
    section: dict[str, Any],
    chart_specs: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    snapshot_offsets: list[dict[str, Any]],
    aspect: str,
) -> list[ChartArtifact]:
    artifacts: list[ChartArtifact] = []
    for chart_id in section.get("charts", []):
        spec = chart_specs.get(chart_id)
        if spec is None:
            artifacts.append(build_seasonal_chart(pd.Series(dtype=float), title=f"{chart_id}: 数据缺失/未找到配置", chart_id=str(chart_id), aspect=aspect))
            continue
        artifacts.append(_chart_from_spec(chart_id, spec, frames, snapshot_offsets, aspect))
    return artifacts


def _render_weekly_section(
    section: dict[str, Any],
    artifacts: list[ChartArtifact],
    output_dir: str,
    aspect: str,
) -> None:
    title = section.get("title", section.get("id", ""))
    st.markdown(f"#### {title}")
    if artifacts:
        if st.button("复制本板块", key=f"copy_section_{section.get('id', title)}"):
            ok, message = _copy_section_to_clipboard(artifacts, section, aspect)
            if ok:
                st.success(message + "，可直接 Ctrl+V 粘贴。")
            else:
                st.error(message)
    section_key = str(section.get("id", title))
    for idx, artifact in enumerate(artifacts):
        render_key = f"weekly_{section_key}_{idx}_{artifact.chart_id}"
        _render_chart_card("weekly", artifact, output_dir, aspect, render_key=render_key)
        if idx < len(artifacts) - 1:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


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
        if page_type == "weekly":
            section_artifacts = _build_section_artifacts(section, chart_specs, frames, snapshot_offsets, aspect)
            chart_results.extend(section_artifacts)
            _render_weekly_section(section, section_artifacts, output_dir, aspect)
            continue

        st.markdown(f"#### {section.get('title', section.get('id', ''))}")
        if section.get("description"):
            st.caption(str(section.get("description")))
        section_key = str(section.get("id", "section"))
        for chart_idx, chart_id in enumerate(section.get("charts", [])):
            spec = chart_specs.get(chart_id)
            if spec is None:
                st.info(f"{chart_id}: 数据缺失/未找到配置")
                continue
            artifact = _chart_from_spec(chart_id, spec, frames, snapshot_offsets, aspect)
            chart_results.append(artifact)
            render_key = f"{page_type}_{section_key}_{chart_idx}_{chart_id}"
            _render_chart_card(page_type, artifact, output_dir, aspect, render_key=render_key)

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

"""
Export Metrics to Excel/CSV — Thesis Report.

Columns exported (one row per patient):
  Clinical: Mean Shift, Median Shift, P95 Shift, Max Shift (mm), Near Cavity Mean,
            Far Field Mean, Folding Voxels, Roughness Score
  Quality:  Rigid NCC, Rigid Score, Rigid Verdict

Produces:
  brain_shift_statistics.xlsx  — formatted Excel with Legend sheet
  brain_shift_statistics.csv   — plain CSV backup
"""

import re
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))
from src.core import setup_logger, load_config
from src.data_manager import DataManager

logger = setup_logger("export_metrics")

# ---------------------------------------------------------------------------
# Column definitions: (excel_header, description, format)
# format: 'mm2' = 2 decimal float in mm, 'int' = integer, 'pct' = 0-1→%, 'str'
# ---------------------------------------------------------------------------
COLUMNS = [
    # Identity
    ("Patient",           "Patient identifier",                                             "str"),
    # Brain shift magnitude
    ("Mean Shift (mm)",   "Mean displacement across all brain voxels (mm)",                 "mm2"),
    ("Median Shift (mm)", "Median displacement — robust to outliers (mm)",                  "mm2"),
    ("P95 Shift (mm)",    "95th percentile displacement — ignores extreme outliers (mm)",   "mm2"),
    ("Max Shift (mm)",    "Maximum displacement anywhere in the brain (mm)",                "mm2"),
    # Spatial analysis
    ("Near Cavity (mm)",  "Mean shift in voxels close to the surgical cavity (mm)",         "mm2"),
    ("Far Field (mm)",    "Mean shift in remote cortex — baseline registration noise (mm)", "mm2"),
    # Deformation quality
    ("Folding (vox)",     "Number of voxels with negative Jacobian (non-physical folding)", "int"),
    ("Roughness",         "Mean gradient magnitude of displacement field — 0 = smooth",     "mm2"),
    # Registration quality
    ("Rigid NCC",         "Normalized Cross-Correlation after rigid alignment (0–1, higher = better)", "mm2"),
    ("Rigid Score",       "Heuristic quality score: 5 = excellent, 1 = poor",               "mm2"),
    ("Rigid Verdict",     "Human-readable registration quality verdict",                     "str"),
    # Robustness (Synthetic Test - Stress Test)
    ("Stability Index", "Success rate of recovery from extreme perturbations (1.0 = rock solid)", "pct"),
    ("Resilience",      "Automated assessment of algorithmic stability",                      "str"),
]

LEGEND_THRESHOLDS = [
    ("Mean Shift (mm)",   "< 3 mm excellent | 3–8 mm good | > 8 mm poor"),
    ("P95 Shift (mm)",    "< 10 mm excellent | < 20 mm acceptable | > 20 mm suspicious"),
    ("Folding (vox)",     "0 = perfect | < 1000 acceptable | > 10000 = registration artefact"),
    ("Roughness",         "< 0.15 smooth | 0.15–0.30 normal | > 0.30 noisy"),
    ("Rigid NCC",         "> 0.70 excellent | 0.55–0.70 good | < 0.55 marginal"),
    ("Stability Index",   "1.00 = Algorithm converges even with >20mm/10\u00b0 distortions"),
    ("Resilience",        "ROCK SOLID = 100% success | STABLE = >90% | SENSITIVE < 90%"),
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _num(val_str: str):
    """Extract first number from a string, else return None."""
    m = re.search(r"[-+]?\d*\.?\d+", val_str)
    return float(m.group()) if m else None


def parse_kv(file_path: Path) -> dict:
    """Parse 'Key: value' text files into a dict."""
    data = {}
    if not file_path.exists():
        return data
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                n = _num(v)
                data[k] = n if n is not None else v
    return data


def collect_row(pid: str, paths: dict, config) -> dict:
    """Collect all metrics for one patient into a flat dict."""
    out = paths["out_dir"]

    thesis = parse_kv(out / "THESIS_VALIDATION.txt")
    rigid  = parse_kv(out / "VALIDATION_FINAL.txt")

    # Verdict: extract the quality keyword
    verdict_raw = str(rigid.get("Verdict", ""))
    m = re.search(r"\b(EXCELLENT|GOOD|ACCEPTABLE|MARGINAL|POOR)\b", verdict_raw)
    verdict = m.group(1) if m else verdict_raw.replace("[OK]", "").replace("[~]", "").strip() or None

    try:
        import pandas as pd
        # Correct path: figures/robustness_stats/robustness_results.csv (global stats)
        # OR per-patient if they exist
        rob_csv = config.PROJECT_ROOT / "figures" / "robustness_stats" / "robustness_results.csv"
        
        if rob_csv.exists():
            df_rob = pd.read_csv(rob_csv)
            # Find rows for this patient if multiple are present, or just use global if it's a summary
            # A global (multi-case) robustness CSV is filtered by case ID below
            if "Patient" in df_rob.columns:
                p_df = df_rob[df_rob["Patient"] == pid]
                if not p_df.empty:
                    mc_success = p_df["Success_Rate"].iloc[0] if "Success_Rate" in p_df.columns else None
                else:
                    mc_success = None
            else:
                # Fallback to the one we just generated if it was a single run
                num_it = len(df_rob)
                succ = len(df_rob[df_rob["Status"] == "Success"])
                mc_success = succ / num_it if num_it > 0 else None
            
            # Resilience Verdict
            if mc_success is None:  res_verdict = None
            elif mc_success == 1.0: res_verdict = "ROCK SOLID"
            elif mc_success >= 0.9: res_verdict = "STABLE"
            else:                   res_verdict = "SENSITIVE"
        else:
            mc_success, res_verdict = None, None
    except Exception as e:
        logger.debug(f"Robustness parse failed for {pid}: {e}")
        mc_success, res_verdict = None, None

    return {
        "Patient":           pid,
        "Mean Shift (mm)":   thesis.get("Global Mean Shift"),
        "Median Shift (mm)": thesis.get("Global Median Shift"),
        "P95 Shift (mm)":    thesis.get("Global P95 Shift"),
        "Max Shift (mm)":    thesis.get("Global Max Shift"),
        "Near Cavity (mm)":  thesis.get("Near Cavity Mean"),
        "Far Field (mm)":    thesis.get("Far Field Mean"),
        "Folding (vox)":     thesis.get("Folding (Neg Jacobian)"),
        "Roughness":         thesis.get("Roughness Score"),
        "Rigid NCC":         rigid.get("Global NCC"),
        "Rigid Score":       rigid.get("Score"),
        "Rigid Verdict":     verdict or None,
        "Stability Index":   mc_success,
        "Resilience":        res_verdict,
    }


# ---------------------------------------------------------------------------
# Excel formatting
# ---------------------------------------------------------------------------

def _format_excel(df, path: Path):
    """Write df to xlsx with professional formatting + Legend sheet."""
    try:
        import openpyxl
        from openpyxl.styles import (PatternFill, Font, Alignment,
                                     Border, Side)
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.warning("openpyxl not installed — saving plain xlsx via pandas fallback.")
        df.to_excel(path, index=False)
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Brain Shift Data"

    # ── Styles ──
    HEADER_FILL  = PatternFill("solid", fgColor="1E3A5F")   # dark navy
    HEADER_FONT  = Font(color="FFFFFF", bold=True, size=10)
    ALT_FILL     = PatternFill("solid", fgColor="EDF4FF")   # light blue stripe
    CENTER       = Alignment(horizontal="center", vertical="center", wrap_text=True)
    LEFT         = Alignment(horizontal="left",   vertical="center")
    THIN_BORDER  = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    col_meta = {c[0]: c for c in COLUMNS}

    # ── Header row ──
    headers = list(df.columns)
    ws.row_dimensions[1].height = 36
    for ci, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.fill        = HEADER_FILL
        cell.font        = HEADER_FONT
        cell.alignment   = CENTER
        cell.border      = THIN_BORDER

    # ── Data rows ──
    col_formats = {h: col_meta.get(h, (h, "", "str"))[2] for h in headers}

    for ri, (_, row) in enumerate(df.iterrows(), start=2):
        fill = ALT_FILL if ri % 2 == 0 else None
        ws.row_dimensions[ri].height = 18
        for ci, h in enumerate(headers, start=1):
            val = row[h]
            fmt = col_formats[h]

            # Format value
            if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "float"
                               and str(val) == "nan"):
                display = None
            elif fmt == "mm2":
                try:
                    display = round(float(val), 2)
                except (TypeError, ValueError):
                    display = val
            elif fmt == "int":
                try:
                    display = int(float(val))
                except (TypeError, ValueError):
                    display = val
            else:
                display = val

            cell = ws.cell(row=ri, column=ci, value=display)
            cell.border    = THIN_BORDER
            cell.alignment = LEFT if fmt == "str" else CENTER
            if fill:
                cell.fill = fill

    # ── Column widths ──
    width_map = {
        "Patient":           16,
        "Mean Shift (mm)":   14,
        "Median Shift (mm)": 15,
        "P95 Shift (mm)":    14,
        "Max Shift (mm)":    14,
        "Near Cavity (mm)":  16,
        "Far Field (mm)":    14,
        "Folding (vox)":     14,
        "Roughness":         12,
        "Rigid NCC":         12,
        "Rigid Score":       12,
        "Rigid Verdict":     24,
    }
    for ci, h in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = width_map.get(h, 14)

    # Freeze top row + Patient column
    ws.freeze_panes = "B2"

    # ── Legend sheet ──
    wl = wb.create_sheet("Legend")
    wl.column_dimensions["A"].width = 22
    wl.column_dimensions["B"].width = 55
    wl.column_dimensions["C"].width = 50

    wl.cell(1, 1, "Column").font        = Font(bold=True, size=10)
    wl.cell(1, 2, "Description").font   = Font(bold=True, size=10)
    wl.cell(1, 3, "Thresholds").font    = Font(bold=True, size=10)
    for ci in (1, 2, 3):
        c = wl.cell(1, ci)
        c.fill      = HEADER_FILL
        c.font      = Font(color="FFFFFF", bold=True, size=10)
        c.alignment = CENTER
        c.border    = THIN_BORDER
    wl.row_dimensions[1].height = 22

    thresh_map = dict(LEGEND_THRESHOLDS)

    for ri2, (col_name, desc, _) in enumerate(COLUMNS, start=2):
        wl.row_dimensions[ri2].height = 20
        fill2 = ALT_FILL if ri2 % 2 == 0 else None
        for ci, val in enumerate([col_name, desc, thresh_map.get(col_name, "—")], start=1):
            c = wl.cell(ri2, ci, val)
            c.alignment = Alignment(vertical="center", wrap_text=True)
            c.border    = THIN_BORDER
            if fill2:
                c.fill = fill2

    wb.save(path)
    logger.success(f"Saved formatted Excel: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        config = load_config()
    except Exception as e:
        print(f"Config error: {e}")
        return

    dm       = DataManager(config)
    patients = dm.get_available_patients()

    if not patients:
        logger.warning("No processed patients found.")
        return

    logger.info(f"Exporting {len(patients)} patients...")

    rows = []
    for pid in patients:
        paths = dm.get_patient_paths(pid)
        row   = collect_row(pid, paths, config)
        rows.append(row)

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Run: pip install pandas openpyxl")
        return

    df = pd.DataFrame(rows, columns=[c[0] for c in COLUMNS])

    # Drop rows that are entirely empty (except Patient)
    metric_cols = [c[0] for c in COLUMNS if c[0] != "Patient"]
    df = df.dropna(subset=metric_cols, how="all")

    # Sort by patient ID
    df = df.sort_values("Patient").reset_index(drop=True)

    # Save Excel
    xlsx_path = Path("brain_shift_statistics.xlsx")
    _format_excel(df, xlsx_path)

    # Save CSV backup
    csv_path = Path("brain_shift_statistics.csv")
    df.to_csv(csv_path, index=False)
    logger.success(f"Saved CSV: {csv_path}")

    # Console summary
    print(f"\n{'='*70}")
    print(f"  Patients exported : {len(df)}")
    print(f"  Excel             : {xlsx_path.absolute()}")
    print(f"  CSV               : {csv_path.absolute()}")
    print(f"{'='*70}\n")
    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.max_columns", None,
                           "display.width", 120):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

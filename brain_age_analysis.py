"""
Cleaned and commented analysis script for Brain Age comparisons.This script will allow you to replicate the main results of [...]


Author: Alejandro Roig Herrero LLM assisted
Date: 2025-09-29

English
USAGE
-----
This script is ready to be use with the excel data in Mendeley Data doi: 10.17632/nz2wnz3vhk.1.
modify the routes at the end of this script:
if __name__ == "__main__":
    excel_path = r"path/to/the/excelfile.xlsx" #the r before the string is in case you are using windows.
    save_plot = r"path_if_you_want_images_to_be_saved.png"

    run_pipeline(excel_path, save_plot)

NOTES
-----
- The script prints model summaries and statistics to stdout.
- Figures are shown on screen and (optionally) saved if --save-plot is provided.
- IMPORTANTLY: if you want to do the analyses in the Pyment model resutls (deep learning) you must:
    Change: Predicted_corrected -> Pyment_prediction
    Change: GAP_corrected -> Pyment_GAP
    Note that Pyment has no corrected version, therefore results including comparisons between corrected vs uncorrected version of the model
    should be ignored when re-running the pyment version.

Español
USO
---
Este script está listo para usarse con el conjunto de datos en Excel disponible en Mendeley Data (DOI: 10.17632/nz2wnz3vhk.1).  
Modifica las rutas al final de este script:

if __name__ == "__main__":
    excel_path = r"ruta/al/archivo_excel.xlsx"   # la 'r' al inicio es recomendable cuando se usan rutas en Windows
    save_plot = r"ruta/si_quieres_guardar_las_imagenes.png"

    run_pipeline(excel_path, save_plot)

NOTAS
-----
- El script imprime en pantalla los resúmenes de los modelos y las estadísticas.
- Las figuras se muestran en pantalla y (opcionalmente) se guardan si se especifica `save_plot`.
- IMPORTANTE: si deseas ejecutar los análisis con los resultados del modelo Pyment (deep learning), debes:
    * Sustituir `Predicted_corrected` por `Pyment_prediction`
    * Sustituir `GAP_corrected` por `Pyment_GAP`
  Ten en cuenta que Pyment no proporciona una versión corregida. Por lo tanto, los resultados que incluyan comparaciones entre versiones corregidas y sin corregir deben ignorarse al volver a ejecutar los análisis con el modelo Pyment.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler
from scipy import stats

# ------------------------------
# Matplotlib global style (+2 points vs defaults, as requested)
# ------------------------------
plt.rcParams.update({
    'font.size': 16,             # default 10
    'axes.labelsize': 18,        # default 12
    'axes.titlesize': 18,        # default 12
    'xtick.labelsize': 14,       # default 10
    'ytick.labelsize': 14,       # default 10
    'legend.fontsize': 14,       # default 10
    'figure.titlesize': 20       # default 12
})


# ------------------------------
# Helper dataclasses and functions
# ------------------------------
@dataclass
class EffectSizes:
    r2: float
    f2: float


def compute_f2(r2: float) -> float:
    """Cohen's f^2 from R^2."""
    denom = (1.0 - r2)
    return (r2 / denom) if denom != 0 else np.nan


def partial_effect(full_model, reduced_model) -> Tuple[float, float]:
    """
    Compute ΔR² and Δf² for a term by comparing a full vs reduced model.
    f² uses the full model's R² in the denominator to match common practice.
    """
    delta_r2 = full_model.rsquared - reduced_model.rsquared
    denom = (1.0 - full_model.rsquared)
    delta_f2 = (delta_r2 / denom) if denom != 0 else np.nan
    return delta_r2, delta_f2


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return a boolean mask for outliers using 1.5*IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def zscore_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Z-score specified columns, in place, returning df for chaining."""
    scaler = StandardScaler()
    df[list(cols)] = scaler.fit_transform(df[list(cols)])
    return df


def describe_after_preprocessing(df: pd.DataFrame) -> None:
    """Print descriptive statistics after preprocessing."""
    print("\n=== Descriptive statistics (post-preprocessing) ===")
    print(df.describe(include="all"))


def fit_ols_robust(formula: str, data: pd.DataFrame):
    """Fit OLS with HC3 robust covariance and return the fitted model."""
    return smf.ols(formula, data=data).fit(cov_type='HC3')


def print_model_summary(title: str, model) -> EffectSizes:
    """Print a header + model summary + global effect size, return EffectSizes."""
    print(f"\n=== {title} ===")
    print(model.summary())
    r2 = model.rsquared
    f2 = compute_f2(r2)
    print(f"\nGlobal model: R² = {r2:.4f}, Cohen's f² = {f2:.4f}")
    return EffectSizes(r2=r2, f2=f2)


def adjusted_outcome(df: pd.DataFrame, model, covariates: List[str], y_col: str) -> pd.Series:
    """
    Compute adjusted outcome by removing linear contributions of covariates
    (at the mean of those covariates).
    """
    means = df[covariates].mean()
    adjustment = np.zeros(len(df))
    for var in covariates:
        if var in model.params.index:
            adjustment += model.params[var] * (df[var] - means[var])
    return df[y_col] - adjustment


def mannwhitney_effects(u_stat: float, n1: int, n2: int) -> Tuple[float, float]:
    """
    Return (r, rank-biserial) effect sizes for a Mann-Whitney U.
    r = Z / sqrt(n1+n2)
    rank-biserial r_B = 1 - (2U)/(n1*n2)
    """
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u_stat - mu_u) / sigma_u if sigma_u != 0 else np.nan
    r = z / np.sqrt(n1 + n2) if (n1 + n2) > 0 else np.nan
    r_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else np.nan
    return r, r_biserial


def plot_control_age_fit(df_ctrl: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot chronological vs corrected predicted age for controls.
    Uses pure matplotlib (no seaborn), with regression line and y=x line.
    """
    x = df_ctrl['Age'].to_numpy()
    y = df_ctrl['Predicted_corrected'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.7, label="Controls")

    # OLS fit line via numpy.polyfit (1st-degree)
    if len(x) > 1 and len(y) > 1:
        b1, b0 = np.polyfit(x, y, 1)  # y ≈ b1*x + b0
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        ax.plot(x_line, y_line, linewidth=2, label="Regression line")
    else:
        print("Not enough points to draw a regression line.")

    # Perfect correlation line y = x
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, linestyle='--', label="Perfect correlation")

    # Axis limits and ticks
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    dy = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    ax.set_xlabel("Chronological age")
    ax.set_ylabel("Corrected predicted age")
    ax.legend()
    plt.tight_layout(pad=0.1)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {out.resolve()}")
    plt.show()


# ------------------------------
# Main analysis pipeline
# ------------------------------
def run_pipeline(excel_path: Path, save_plot: str | None) -> None:
    # 1) Load data
    df = pd.read_excel(excel_path)
    print(f"Loaded data from: {excel_path} (n={len(df)})")

    # 2) Outlier detection on GAP_corrected (no removal, just report)
    outliers_mask = detect_outliers_iqr(df['GAP_corrected'])
    df_outliers = df.loc[outliers_mask, ['codigo', 'GAP_corrected']]
    print("\nOutlier codes in 'GAP_corrected' (IQR method):")
    print(df_outliers['codigo'].dropna().unique())

    # Visual check via boxplot
    plt.figure(figsize=(8, 4))
    plt.boxplot(df['GAP_corrected'], vert=False)
    plt.title("Boxplot of 'GAP_corrected' for outlier detection")
    plt.xlabel("GAP_corrected")
    plt.tight_layout()
    plt.show()

    # Means of GAP_corrected across groups (as in the original)
    mean_control = df[df['Group'] == 'Control']['GAP_corrected'].mean()
    mean_sqzfr = df[df['Group'].isin(['SQZFR_cron', 'SQZFR_primer_ep'])]['GAP_corrected'].mean()
    mean_ep = df[df['Group'] == 'SQZFR_primer_ep']['GAP_corrected'].mean()
    print(f"\nMean GAP_corrected (Control): {mean_control:.4f}")
    print(f"Mean GAP_corrected (SQZFR)  : {mean_sqzfr:.4f}")
    print(f"Mean GAP_corrected (EP)     : {mean_ep:.4f}")

    # 3) Standardize continuous variables (mimics original script)
    continuous_vars = ['Age', 'eTIV', 'BMI', 'GAP_corrected', 'duration', 'equiv_cpz']
    missing = [c for c in continuous_vars if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected continuous columns: {missing}")
    df = zscore_columns(df, continuous_vars)

    # 4) Cast categoricals (Group is required; bd_ap is optional)
    df['Group'] = df['Group'].astype('category')
    if 'bd_ap' in df.columns:
        df['bd_ap'] = df['bd_ap'].astype('category')

    # 5) Descriptive stats
    describe_after_preprocessing(df)

    # 6) Group comparison: Control vs SQZFR (SQZFR_cron + SQZFR_primer_ep)
    df_comp1 = df[df['Group'].isin(['Control', 'SQZFR_cron', 'SQZFR_primer_ep'])].copy()
    df_comp1['Grupo'] = df_comp1['Group'].apply(lambda x: 'Control' if x == 'Control' else 'SQZFR')
    formula_comp1 = "GAP_corrected ~ C(Grupo) + Age + sex_M_1_F_0_ + eTIV + BMI"
    model_comp1 = fit_ols_robust(formula_comp1, df_comp1)
    print("\n=== Comparison 1: Control vs. SQZFR (all) – OLS robust ===")
    # print only p-value for the group effect, as original code did
    try:
        p_val = model_comp1.pvalues["C(Grupo)[T.SQZFR]"]
        print(f"p-value for SQZFR vs Control: {p_val:.10f}")
    except KeyError:
        print("Term C(Grupo)[T.SQZFR] not found in model parameters.")
    r2_1 = model_comp1.rsquared
    f2_1 = compute_f2(r2_1)
    print(f"Global model: R² = {r2_1:.4f}, Cohen's f² = {f2_1:.4f}")

    # 7) Comparison: Control vs SQZFR_cron
    df_cron = df[df['Group'].isin(['Control', 'SQZFR_cron'])].copy()
    df_cron['Grupo'] = df_cron['Group'].apply(lambda x: 'Control' if x == 'Control' else 'SQZFR_cron')
    formula_cron = "GAP_corrected ~ C(Grupo) + Age + sex_M_1_F_0_ + eTIV + BMI"
    model_cron = fit_ols_robust(formula_cron, df_cron)
    print_model_summary("Comparison: Control vs. SQZFR_cron – OLS robust", model_cron)

    # Partial effect of 'Grupo' in comp1 (compare to reduced model without group)
    formula_comp1_red = "GAP_corrected ~ Age + sex_M_1_F_0_ + eTIV + BMI"
    model_comp1_red = fit_ols_robust(formula_comp1_red, df_comp1)
    delta_r2_1, delta_f2_1 = partial_effect(model_comp1, model_comp1_red)
    print(f"Partial effect of 'Grupo' (Comp1): ΔR² = {delta_r2_1:.4f}, ΔCohen's f² = {delta_f2_1:.4f}")

    # 9) Comparison 2: Control vs SQZFR_primer_ep
    df_comp2 = df[df['Group'].isin(['Control', 'SQZFR_primer_ep'])].copy()
    # Clean categories to avoid unused
    df_comp2['Group'] = df_comp2['Group'].cat.remove_unused_categories()
    formula_comp2 = "GAP_corrected ~ C(Group) + Age + sex_M_1_F_0_ + eTIV + BMI"
    model_comp2 = fit_ols_robust(formula_comp2, df_comp2)
    print_model_summary("Comparison 2: Control vs. SQZFR_primer_ep – OLS robust", model_comp2)

    # Post-hoc t-tests on adjusted outcome (Welch) across levels of Group in comp2
    covariates = ['Age', 'sex_M_1_F_0_', 'eTIV', 'BMI']
    df_comp2['GAP_corrected_adj'] = adjusted_outcome(df_comp2, model_comp2, covariates, 'GAP_corrected')

    levels = list(df_comp2['Group'].unique())
    from itertools import combinations
    pairs = list(combinations(levels, 2))
    print("\n=== Post-hoc comparisons for Group (uncorrected and Bonferroni-corrected) ===")
    pvals = []
    for g1, g2 in pairs:
        d1 = df_comp2.loc[df_comp2['Group'] == g1, 'GAP_corrected_adj']
        d2 = df_comp2.loc[df_comp2['Group'] == g2, 'GAP_corrected_adj']
        t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=False)
        print(f"{g1} vs {g2}: t = {t_stat:.4f}, p (uncorr) = {p_val:.4f}")
        pvals.append(p_val)

    if pvals:
        reject, pvals_corr, _, _ = multipletests(pvals, method='bonferroni')
        print("\nBonferroni-corrected p-values:")
        for (g1, g2), p_corr in zip(pairs, pvals_corr):
            print(f"{g1} vs {g2}: p (corr) = {p_corr:.4f}")

    # Partial effect of 'Group' in comp2
    formula_comp2_red = "GAP_corrected ~ Age + sex_M_1_F_0_ + eTIV + BMI"
    model_comp2_red = fit_ols_robust(formula_comp2_red, df_comp2)
    d_r2, d_f2 = partial_effect(model_comp2, model_comp2_red)
    print(f"Partial effect of Group (Comp2): ΔR² = {d_r2:.4f}, ΔCohen's f² = {d_f2:.4f}")

    # 10) Medication analysis in SQZFR_cron
    df_pat = df[df['Group'].isin(['SQZFR_cron'])].copy()
    df_pat = df_pat.dropna(subset=['equiv_cpz'])
    if not df_pat.empty:
        df_pat_b = df_pat.dropna(subset=['duration']).copy()
        if not df_pat_b.empty:
            # 10.4 Interaction: equiv_cpz * duration
            formula_med_int = "GAP_corrected ~ equiv_cpz * duration + Age + sex_M_1_F_0_ + eTIV + BMI"
            model_med_int = fit_ols_robust(formula_med_int, df_pat_b)
            print_model_summary("Medication Model Int: equiv_cpz * duration", model_med_int)

            formula_med_int_red = "GAP_corrected ~ equiv_cpz + duration + Age + sex_M_1_F_0_ + eTIV + BMI"
            model_med_int_red = fit_ols_robust(formula_med_int_red, df_pat_b)
            d_r2_int, d_f2_int = partial_effect(model_med_int, model_med_int_red)
            print(f"Partial effect of interaction (equiv_cpz:duration): ΔR² = {d_r2_int:.4f}, ΔCohen's f² = {d_f2_int:.4f}")
        else:
            print("Skipping Medication Model B/Interaction: no non-missing 'duration' in SQZFR_cron.")
    else:
        print("Skipping medication analyses: no non-missing 'equiv_cpz' in SQZFR_cron.")

    # 12) Non-parametric comparison for bd_ap (Yes/No) using original-scale bd_ap for labeling
    # Reload original (unscaled) dataframe for this part to mirror original behavior
    df_bd = pd.read_excel(excel_path)
    if 'bd_ap' in df_bd.columns:
        # Map to human labels if bd_ap is 0/1. Keep as-is otherwise.
        try:
            df_bd['bd_ap'] = df_bd['bd_ap'].astype('category').cat.rename_categories({0: 'No', 1: 'Si'})
        except ValueError:
            df_bd['bd_ap'] = df_bd['bd_ap'].astype('category')

        group_ap = df_bd[df_bd['bd_ap'] == "Si"]['GAP_corrected'].dropna()
        group_no = df_bd[df_bd['bd_ap'] == "No"]['GAP_corrected'].dropna()

        # Mann-Whitney U (two-sided)
        if not group_ap.empty and not group_no.empty:
            u_stat, p_val = stats.mannwhitneyu(group_ap, group_no, alternative='two-sided')
            print("\nNON-PARAMETRIC COMPARISON (Bipolar bd_ap: Yes vs No)")
            print(f"Mann-Whitney U = {u_stat:.3f}, p = {p_val:.4f}")
            print(f"Means → No AP: {group_no.mean():.4f}, Yes AP: {group_ap.mean():.4f}")

            r, r_biserial = mannwhitney_effects(u_stat, len(group_ap), len(group_no))
            print(f"Effect size r (from Z): {r:.2f}")
            print(f"Rank-biserial correlation r_B: {r_biserial:.2f}")

            # Descriptive: mean/median
            print("\nDescriptive statistics")
            print(f"AP Yes: mean = {group_ap.mean():.4f}, median = {group_ap.median():.4f}")
            print(f"AP No : mean = {group_no.mean():.4f}, median = {group_no.median():.4f}")
        else:
            print("Skipping Mann-Whitney: one of the groups (Yes/No) is empty.")
    else:
        print("Skipping Mann-Whitney: 'bd_ap' column not found.")

    # 13) Control group: prediction metrics before/after correction + plot
    df_ctrl = df[df['Group'] == 'Control'].copy()
    # For metrics, we want original-scale 'Predicted' and 'Predicted_corrected'
    df_ctrl_raw = pd.read_excel(excel_path)
    df_ctrl_raw = df_ctrl_raw[df_ctrl_raw['Group'] == 'Control']
    # Check required columns exist
    for col in ['Age', 'Predicted', 'Predicted_corrected']:
        if col not in df_ctrl_raw.columns:
            raise KeyError(f"Column '{col}' required for control metrics is missing in the Excel file.")

    from sklearn.metrics import mean_absolute_error
    from scipy.stats import pearsonr

    # Uncorrected
    mae_u = mean_absolute_error(df_ctrl_raw['Age'], df_ctrl_raw['Predicted'])
    r_u, p_u = pearsonr(df_ctrl_raw['Age'], df_ctrl_raw['Predicted'])
    print(f"\nControl metrics (uncorrected): MAE = {mae_u:.4f}, r = {r_u:.4f} (p = {p_u:.4f})")

    # Corrected
    mae_c = mean_absolute_error(df_ctrl_raw['Age'], df_ctrl_raw['Predicted_corrected'])
    r_c, p_c = pearsonr(df_ctrl_raw['Age'], df_ctrl_raw['Predicted_corrected'])
    print(f"Control metrics (corrected):   MAE = {mae_c:.4f}, r = {r_c:.4f} (p = {p_c:.4f})")

    # Plot for controls (Age vs Predicted_corrected)
    if not df_ctrl_raw.empty:
        plot_control_age_fit(df_ctrl_raw[['Age', 'Predicted_corrected']].assign(Group='Control'), save_plot)
    else:
        print("Skipping control age-fit plot: no controls found.")

    print("\n=== DONE ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleaned, modular Brain Age analysis script (replicates original behavior)."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        required=True,
        help="Path to the Excel file (e.g., base_definitiva_armo_corrected.xlsx)"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to save the control age-fit PNG (e.g., ./control_age_fit.png)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    excel_path = r"path/to/the/excelfile.xlsx" #the r before the string is in case you are using windows.
    save_plot = r"path_if_you_want_images_to_be_saved.png" #the r before the string is in case you are using windows.

    run_pipeline(excel_path, save_plot)


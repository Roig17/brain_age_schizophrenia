"""
Cleaned and commented analysis script for Brain Age comparisons.This script will allow you to replicate the
assumptions of the main analyses of [...]


Author: Alejandro Roig Herrero LLM assisted
Date: 2025-09-29
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler

# Ajustes globales de Matplotlib con +2 puntos sobre los defaults
plt.rcParams.update({
    'font.size': 16,             # default es 10
    'axes.labelsize': 18,        # default es 12
    'axes.titlesize': 18,        # default es 12
    'xtick.labelsize': 14,       # default es 10
    'ytick.labelsize': 14,       # default es 10
    'legend.fontsize': 14,       # default es 10
    'figure.titlesize': 20       # default es 12
})



# Cargar el DataFrame desde el archivo Excel
archivo = r"path_to_excel_file.xlsx"
df = pd.read_excel(archivo)

import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from tabulate import tabulate  # pip install tabulate

def print_assumption_results(name, results):
    """
    Pretty print for check_model_assumptions results.
    """
    print("\n" + "="*60)
    print(f" Assumption checks for: {name} ")
    print("="*60)

    # 1. Normality
    norm = results.get("normality")
    if norm:
        print("\n[Normality of residuals – Shapiro-Wilk]")
        print(tabulate([[norm['shapiro_stat'], norm['p_value']]],
                       headers=["W-statistic", "p-value"],
                       tablefmt="grid"))

    # 2. Homocedasticity
    homo = results.get("homocedasticity")
    if homo:
        print("\n[Homoscedasticity – Breusch–Pagan]")
        print(tabulate([[homo['LM_statistic'], homo['LM_p_value'],
                         homo['F_statistic'], homo['F_p_value']]],
                       headers=["LM stat", "LM p", "F stat", "F p"],
                       tablefmt="grid"))

    # 3. Residuals by group
    resid_by_group = results.get("resid_by_group")
    if resid_by_group:
        print("\n[Residuals by group (mean ≈ 0 check)]")
        rows = []
        for g, vals in resid_by_group.items():
            rows.append([g, vals['mean'], vals['t_statistic'], vals['p_value']])
        print(tabulate(rows, headers=["Group", "Mean", "t-stat", "p-value"], tablefmt="grid"))

    # 4. Homogeneity of slopes
    slope = results.get("slope_homogeneity")
    if slope:
        print("\n[Homogeneity of slopes (interaction vs additive model)]")
        print(tabulate([[slope['F_statistic'], slope['p_value'], slope['df_diff']]],
                       headers=["F stat", "p-value", "df diff"],
                       tablefmt="grid"))


def check_model_assumptions(modelo, df, group_var=None, predictors=None):
    """
    English
    Evaluates several assumptions of a linear model:
      1. Zero-mean residuals within each group (if group_var is specified).
      2. Normality of residuals (Shapiro–Wilk test, global).
      3. Homoscedasticity of residuals (Breusch–Pagan test, global).
      4. Homogeneity of within-group slopes, by comparing an interaction model
         vs. an additive model (if group_var and predictors are specified).

    Parameters:
      modelo: Fitted statsmodels object (result of smf.ols(...).fit()).
      df: DataFrame used to fit the model.
      group_var (str, optional): Name of the grouping variable.
      predictors (list, optional): List of predictor variable names (excluding the group term)
                                   that were used in the additive model.
    
    Returns:
      results (dict): Dictionary with the following elements:
            - 'resid_by_group': dict with mean, t-statistic, and p-value for each group (if group_var is specified).
            - 'normality': dict with statistic and p-value of the Shapiro–Wilk test.
            - 'homoscedasticity': dict with LM statistic, LM p-value, F-statistic, and F-test p-value (Breusch–Pagan).
            - 'slope_homogeneity': dict with F, p-value, and degrees of freedom of the slope homogeneity test
                                   (if group_var and predictors are specified); otherwise, None.

    Español
    Evalúa varias asunciones de un modelo lineal:
      1. Media cero de los residuos en cada grupo (si se especifica group_var).
      2. Normalidad de los residuos (prueba Shapiro–Wilk, globalmente).
      3. Homocedasticidad de los residuos (prueba Breusch–Pagan, globalmente).
      4. Homogeneidad de las within slopes entre grupos, mediante la comparación
         de un modelo con interacciones vs. un modelo aditivo (si se especifica group_var y predictors).

    Parámetros:
      modelo: Objeto ajustado de statsmodels (resultado de smf.ols(...).fit()).
      df: DataFrame usado para ajustar el modelo.
      group_var (str, opcional): Nombre de la variable de agrupación.
      predictors (list, opcional): Lista de nombres de las variables predictoras (excluyendo el término grupo)
                                    que se usaron en el modelo aditivo.
    
    Retorna:
      results (dict): Diccionario con los siguientes elementos:
            - 'resid_by_group': dict con media, t-est, p-valor para cada grupo (si group_var se especifica).
            - 'normality': dict con estadístico y p-valor del test Shapiro-Wilk.
            - 'homocedasticity': dict con LM statistic, LM p-value, F-statistic y F-test p-value (Breusch–Pagan).
            - 'slope_homogeneity': dict con F, p-valor y grados de libertad del test de homogeneidad de slopes 
                                   (si se especifica group_var y predictors); de lo contrario, None.
    """
    results = {}
    
    # Residuos del modelo
    resid = modelo.resid
    
    # 1. Normalidad de residuos (Shapiro-Wilk, global)
    shapiro_stat, shapiro_p = stats.shapiro(resid)
    results['normality'] = {"shapiro_stat": round(shapiro_stat, 2), "p_value": round(shapiro_p, 5)}
    
    # 2. Homocedasticidad de residuos (Breusch–Pagan)
    bp_test = sm.stats.diagnostic.het_breuschpagan(resid, modelo.model.exog)
    bp_labels = ['LM_statistic', 'LM_p_value', 'F_statistic', 'F_p_value']
    results['homocedasticity'] = {label: round(value, 5) for label, value in zip(bp_labels, bp_test)}
    
    # 3. Media 0 de los residuos por grupo
    if group_var is not None:
        groups = df[group_var].unique()
        resid_by_group = {}
        for g in groups:
            # Para el grupo 'g', calcular la media de los residuos y realizar un t-test contra 0.
            resid_g = modelo.resid[df[group_var] == g]
            t_stat, p_val = stats.ttest_1samp(resid_g, popmean=0)
            resid_by_group[g] = {"mean": round(np.mean(resid_g), 2),
                                  "t_statistic": round(t_stat, 2),
                                  "p_value": round(p_val, 2)}
        results['resid_by_group'] = resid_by_group
    else:
        results['resid_by_group'] = None

    # 4. Homogeneidad de las within slopes:
    slope_homogeneity = None
    if group_var is not None and predictors is not None and len(predictors) > 0:
        pred_str = " + ".join(predictors)
        # Modelo con interacciones: permite que la pendiente varíe entre grupos.
        formula_inter = f"{modelo.model.endog_names} ~ C({group_var}) * ({pred_str})"
        modelo_inter = smf.ols(formula_inter, data=df).fit()
        # Modelo aditivo (sin interacciones)
        formula_red = f"{modelo.model.endog_names} ~ C({group_var}) + {pred_str}"
        modelo_red = smf.ols(formula_red, data=df).fit()
        F_stat, p_val, df_diff = modelo_inter.compare_f_test(modelo_red)
        slope_homogeneity = {"F_statistic": round(F_stat, 2), "p_value": round(p_val, 2), "df_diff": df_diff}
    results['slope_homogeneity'] = slope_homogeneity

    return results

# -----------------------------------------------
# comp1: Control vs. SQZFR (agrupando SQZFR_cron y SQZFR_primer_ep)
# -----------------------------------------------
df_comp1 = df[df['Group'].isin(['Control', 'SQZFR_cron', 'SQZFR_primer_ep'])].copy()
# Crear la nueva variable "Grupo": si Group es "Control" se mantiene como "Control", de lo contrario se agrupa como "SQZFR"
df_comp1['Grupo'] = df_comp1['Group'].apply(lambda x: 'Control' if x == 'Control' else 'SQZFR')

print("comp1:")
print(df_comp1[['codigo', 'Group', 'Grupo']].head())
print("\nConteo de grupos en comp1:")
print(df_comp1['Grupo'].value_counts())

# -----------------------------------------------
# comp2: Solo Control y SQZFR_primer_ep
# -----------------------------------------------
df_comp2 = df[df['Group'].isin(['Control', 'SQZFR_primer_ep'])].copy()
print("\ncomp2:")
print(df_comp2[['codigo', 'Group']].head())
print("\nConteo de Group en comp2:")
print(df_comp2['Group'].value_counts())

# -----------------------------------------------
# comp3: Solo SQZFR_cron
# -----------------------------------------------
df_comp3 = df[df['Group'] == 'SQZFR_cron'].copy()
print("\ncomp3:")
print(df_comp3[['codigo', 'Group']].head())
print("\nTotal de SQZFR_cron en comp3:", df_comp3.shape[0])

# -----------------------------------------------
# comp4: Dentro de Bipolar, comparar bd_ap 0 vs. bd_ap 1
# -----------------------------------------------
df_comp4 = df[df['Group'] == 'Bipolar'].copy()
# Asegurarse de que bd_ap tenga datos; eliminar NA en bd_ap
df_comp4 = df_comp4.dropna(subset=['bd_ap']).copy()
# Convertir bd_ap a entero (en caso de que sea categórica) y asignar la etiqueta en función de su valor
df_comp4['Grupo'] = df_comp4['bd_ap'].astype(int).apply(lambda x: 'Bipolar_1' if x == 1 else 'Bipolar_0')

predictors= ['Age', 'BMI', 'eTIV', 'GAP_corrected','sex_M_1_F_0_']
#'duracion', 'equiv_cpz_T1'

formula_example = "GAP_corrected ~ C(Group) + Age + sex_M_1_F_0_ + eTIV + BMI"
modelo_comp1 = smf.ols(formula_example, data=df_comp1).fit(cov_type='HC3')
modelo_comp2 = smf.ols(formula_example, data=df_comp2).fit(cov_type='HC3')
modelo_comp3 = smf.ols(formula_example, data=df_comp3).fit(cov_type='HC3')
modelo_comp4 = smf.ols(formula_example, data=df_comp4).fit(cov_type='HC3')
results_1 = check_model_assumptions(modelo_comp1, df_comp1, group_var="Grupo", predictors=predictors)

results_2 = check_model_assumptions(modelo_comp2, df_comp2, group_var="Group", predictors=predictors)

results_4 = check_model_assumptions(modelo_comp4, df_comp4, group_var="Grupo", predictors=predictors)

results_3 = check_model_assumptions(modelo_comp3, df_comp3, predictors=predictors)


print_assumption_results("comp1 (Control vs SQZFR)", results_1)
print_assumption_results("comp2 (Control vs SQZFR_primer_ep)", results_2)
print_assumption_results("comp3 (SQZFR_cron)", results_3)
print_assumption_results("comp4 (Bipolar bd_ap 0 vs 1)", results_4)

modelo = smf.ols(formula_example, data=df_comp1).fit(cov_type="HC3")

# Obtener residuos
residuos = modelo.resid

# --- Histograma con KDE ---
plt.figure(figsize=(8, 5))
sns.histplot(residuos, kde=True, bins=30, color="skyblue")
plt.title("Histogram of the residuals")
plt.xlabel("Residual")
plt.ylabel("Frecuency")
plt.axvline(0, color="red", linestyle="--")
plt.show()

# --- Q-Q plot ---
sm.qqplot(residuos, line="45", fit=True)
plt.title("Q-Q Plot of residuals")
plt.show()
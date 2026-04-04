import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, pointbiserialr, f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def run_statistical_analysis(df_id):
    """Ejecuta toda la batería de tests estadísticos."""
    results_table = []
    impact_hit = []
    impact_disc = []

    num_vars = ['Distance_Residence_Work', 'Service_time',
                'Age', 'Body_mass_index', 'Son', 'Pet']
    bin_vars = ['Social_drinker', 'Social_smoker']

    for var in num_vars:
        r, p = pearsonr(df_id[var], df_id['Hit_target'])
        impact_hit.append({'Variable': var, 'Impacto': r})
        results_table.append({'Variable': var, 'Test': 'Pearson',
                             'p-value': p, 'Impacto': r, 'Target': 'Hit_target'})

    for var in num_vars:
        df_temp = df_id[[var, 'Disciplinary_failure']].dropna()
        r_b, p = pointbiserialr(df_temp['Disciplinary_failure'], df_temp[var])
        impact_disc.append({'Variable': var, 'Impacto': r_b})
        results_table.append({'Variable': var, 'Test': 'Biserial',
                             'p-value': p, 'Impacto': r_b, 'Target': 'Disciplina'})

    return pd.DataFrame(impact_hit), pd.DataFrame(impact_disc), pd.DataFrame(results_table)


def run_predictive_model(RRHH_full):
    """Entrena el modelo de IA para detectar importancia de variables."""
    features = ['Distance_Residence_Work', 'Service_time', 'Age', 'Son',
                'Pet', 'Body_mass_index', 'Social_drinker', 'Social_smoker']

    X = pd.get_dummies(RRHH_full[features].dropna(), drop_first=True)
    y = RRHH_full.loc[X.index, ['Disciplinary_failure', 'Hit_target']]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    df_imp = pd.DataFrame({
        'Variable': X.columns,
        'Imp_Disc': model.estimators_[0].feature_importances_,
        'Imp_Hit': model.estimators_[1].feature_importances_
    })
    return df_imp

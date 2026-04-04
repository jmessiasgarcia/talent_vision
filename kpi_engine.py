def calculate_general_kpis(df):
    total_abs_hours = df['Absenteeism_hours'].sum()
    total_work_hours = 75168

    tasa_abs = (total_abs_hours / total_work_hours) * 100

    gasto_transporte = df[df['Absenteeism_hours']
                          > 0]['Transportation_expense'].sum()
    bmi_promedio = df['Body_mass_index'].mean()

    return {
        "total_abs_hours": total_abs_hours,
        "tasa_abs": tasa_abs,
        "gasto_transporte": gasto_transporte,
        "bmi_promedio": bmi_promedio
    }

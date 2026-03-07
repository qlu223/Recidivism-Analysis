import pandas as pd

def filter_data():
    df = pd.read_csv("Data/compas-scores-two-years.csv")
    df = df.drop(columns=['id', 'name', 'first', 'last', 'dob', 'compas_screening_date', 'c_case_number', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'decile_score', 'score_text', 'is_recid', 'is_violent_recid', 'event', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'start', 'end', 'c_jail_in', 'c_jail_out', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_desc', 'vr_offense_date', 'screening_date', 'violent_recid', 'decile_score.1', 'priors_count.1', 'decile_score.1', 'days_b_screening_arrest', 'vr_charge_degree'])
    return df

print(filter_data())

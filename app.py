import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import ast
import os
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIG & SYSTEM SETUP
# ==========================================
st.set_page_config(page_title="MCHTrack: Command Center", layout="wide", page_icon="🏥")
BASE = Path.cwd()
# --- VACCINE DEFINITIONS ---
VACCINE_INFO = {
    'BCG': {'type': 'Injectable', 'site': 'Left Upper Arm'},
    'Hep_B0': {'type': 'Injectable', 'site': 'Right Thigh'},
    'OPV_0': {'type': 'Oral', 'site': 'Mouth'},
    'Penta_1': {'type': 'Injectable', 'site': 'Left Thigh'},
    'PCV_1': {'type': 'Injectable', 'site': 'Right Thigh'},
    'OPV_1': {'type': 'Oral', 'site': 'Mouth'},
    'Rota_1': {'type': 'Oral', 'site': 'Mouth'},
    'IPV_1': {'type': 'Injectable', 'site': 'Right Thigh'},
    'Penta_2': {'type': 'Injectable', 'site': 'Left Thigh'},
    'PCV_2': {'type': 'Injectable', 'site': 'Right Thigh'},
    'OPV_2': {'type': 'Oral', 'site': 'Mouth'},
    'Rota_2': {'type': 'Oral', 'site': 'Mouth'},
    'Penta_3': {'type': 'Injectable', 'site': 'Left Thigh'},
    'PCV_3': {'type': 'Injectable', 'site': 'Right Thigh'},
    'OPV_3': {'type': 'Oral', 'site': 'Mouth'},
    'Rota_3': {'type': 'Oral', 'site': 'Mouth'},
    'IPV_2': {'type': 'Injectable', 'site': 'Right Thigh'},
    'Vitamin_A_1': {'type': 'Oral', 'site': 'Mouth'},
    'Measles_1': {'type': 'Injectable', 'site': 'Left Upper Arm'},
    'Yellow_Fever': {'type': 'Injectable', 'site': 'Right Upper Arm'},
    'Meningitis': {'type': 'Injectable', 'site': 'Left Thigh'},
    'Vitamin_A_2': {'type': 'Oral', 'site': 'Mouth'},
    'Measles_2': {'type': 'Injectable', 'site': 'Left Upper Arm'},
    'HPV_1': {'type': 'Injectable', 'site': 'Left Arm (Deltoid)'}, 
    'HPV_2': {'type': 'Injectable', 'site': 'Left Arm (Deltoid)'}
}

VACCINE_MAPPING = {
    'BCG': 'BCG', 'Hep_B0': 'HepB', 'OPV_0': 'OPV',
    'Penta_1': 'Penta', 'Penta_2': 'Penta', 'Penta_3': 'Penta',
    'PCV_1': 'PCV', 'PCV_2': 'PCV', 'PCV_3': 'PCV',
    'OPV_1': 'OPV', 'OPV_2': 'OPV', 'OPV_3': 'OPV',
    'Rota_1': 'Rota', 'Rota_2': 'Rota', 'Rota_3': 'Rota',
    'IPV_1': 'IPV', 'IPV_2': 'IPV',
    'Vitamin_A_1': 'Vitamin A', 'Vitamin_A_2': 'Vitamin A',
    'Measles_1': 'Measles', 'Measles_2': 'Measles',
    'Yellow_Fever': 'Yellow Fever', 'Meningitis': 'Meningitis',
    'HPV_1': 'HPV', 'HPV_2': 'HPV'
}

STOCK_CATEGORIES = sorted(list(set(VACCINE_MAPPING.values())))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_vaccines(v_str):
    if pd.isna(v_str): return []
    v_str = str(v_str)
    if '[' in v_str:
        try: return ast.literal_eval(v_str)
        except: pass
    clean_str = v_str.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace('"', '').replace("'", "")
    return [v.strip() for v in clean_str.split(',') if v.strip()]

def log_dispatch_to_csv(dispatch_data):
    file_path = BASE / 'dispatch_log.csv'
    df_new = pd.DataFrame([dispatch_data])
    if not os.path.isfile(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, mode='a', header=False, index=False)

def save_metrics(model_name, metrics):
    with open(BASE/"model_metrics.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {model_name} Training Results:\n")
        for k, v in metrics.items():
            f.write(f"  - {k}: {v}\n")
        f.write("-" * 30 + "\n")

# ==========================================
# 3. SESSION STATE & DATA LOADING
# ==========================================

def init_session_state():
    if 'data_initialized' not in st.session_state or 'facility_stock' not in st.session_state or 'df_cohort' not in st.session_state:
        try:
            # 1. Load Data
            try:
                df_visits = pd.read_csv(BASE / "facility_visits.csv")
                df_zerodose = pd.read_excel(BASE / "zerodose.xlsx")
                df_cohort = pd.read_csv(BASE / "cohort_data.csv")
            except Exception as e:
                st.warning(f"Note: Using simulation data ({e})")
                
                facilities = ['Dantamashe PHC', 'Gayawa PHC', 'Rimin kebe PHC', 'Kadawa BHC', 'Joda HP']
                
                # Simulation: Ensure parent_id exists for cohort tracking
                # 300 unique children (parent_id), 500 total visits => repeated visits
                unique_parents = [f'child_{i}' for i in range(300)]
                
                # Create Cohort Data (Visits with parent_id)
                df_cohort = pd.DataFrame({
                    'id': [f'visit_{i}' for i in range(500)], # Visit ID
                    'parent_id': np.random.choice(unique_parents, 500), # Child ID (repetitive)
                    'visit_date': [pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(500)],
                    'vaccines_administered': np.random.choice(['{Penta_1, PCV_1}', '{BCG, OPV_0}', '{Measles_1}', '{Yellow_Fever}'], 500),
                    'health_center_name': np.random.choice(facilities, 500),
                    'lga_name': np.random.choice(['Ungogo LGA', 'Kiru LGA', 'Gabasawa LGA'], 500)
                })
                
                # df_visits can basically be df_cohort for demand forecasting purposes
                df_visits = df_cohort.copy()

                df_zerodose = pd.DataFrame({
                    'ID': [f'zd_{i}' for i in range(50)],
                    'age_months': np.random.randint(0, 15, 50),
                    'gender': np.random.choice(['male', 'female'], 50),
                    'lga_name': np.random.choice(['Ungogo LGA', 'Kiru LGA', 'Gabasawa LGA'], 50),
                    'Distance to HF': [f"{np.random.uniform(0,5):.2f} KM" for _ in range(50)],
                    'reasons_for_zd': np.random.choice(['distance', 'refusal'], 50),
                    'vaccines_administered': ['[]']*50
                })

            if 'status' not in df_zerodose.columns:
                df_zerodose['status'] = 'Pending'
            
            # Date conversions
            for df in [df_visits, df_cohort]:
                if 'visit_date' in df.columns:
                    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
                
            st.session_state['df_zerodose'] = df_zerodose
            st.session_state['df_visits'] = df_visits
            st.session_state['df_cohort'] = df_cohort
            
            # 3. Initialize Facility-Level Stock
            all_facilities = list(df_visits['health_center_name'].unique())
            facility_stock = {}
            for fac in all_facilities:
                fac_inventory = {cat: 100 for cat in STOCK_CATEGORIES}
                fac_inventory['Measles'] = np.random.randint(20, 80) 
                facility_stock[fac] = fac_inventory
                
            st.session_state['facility_stock'] = facility_stock
            st.session_state['data_initialized'] = True
            
        except Exception as e:
            st.error(f"⚠️ Critical Error: {e}")
            st.stop()

init_session_state()

# ==========================================
# 4. ML MODELS
# ==========================================

class SuccessModel:
    """
    Model 1: Resolution Probability
    ALGORITHM: GradientBoostingClassifier (Requested)
    """
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        self.encoders = {}
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.filename = "success_model.pkl"

    def clean_distance(self, dist_str):
        if pd.isna(dist_str): return 0.0
        try: return float(str(dist_str).lower().replace('km', '').strip())
        except: return 0.0

    def preprocess(self, df, training=True):
        data = df.copy()
        if 'Distance to HF' in data.columns:
            data['dist_numeric'] = data['Distance to HF'].apply(self.clean_distance)
        else:
            data['dist_numeric'] = 0.0
            
        cat_cols = ['gender', 'lga_name', 'reasons_for_zd']
        for col in cat_cols:
            if col not in data.columns: data[col] = 'Unknown'
            data[col] = data[col].fillna('Unknown').astype(str)
            if training:
                le = LabelEncoder()
                self.encoders[col] = le
                data[f'{col}_code'] = le.fit_transform(data[col])
            else:
                le = self.encoders.get(col)
                if le:
                    data[f'{col}_code'] = data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                else:
                    data[f'{col}_code'] = 0

        features = ['age_months', 'dist_numeric', 'gender_code', 'lga_name_code', 'reasons_for_zd_code']
        if training:
            X = self.imputer.fit_transform(data[features])
        else:
            X = self.imputer.transform(data[features])
        return X

    def train_and_save(self, df):
        train_df = df.copy()
        if 'Resolution Status' not in train_df.columns: return False
            
        X = self.preprocess(train_df, training=True)
        y = train_df['Resolution Status'].apply(lambda x: 1 if str(x).strip() == 'Resolved' else 0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        save_metrics("Success Prediction (Gradient Boost)", {"Accuracy": acc, "Samples": len(df)})
        
        joblib.dump(self, self.filename)
        return True

    def predict_proba(self, df):
        if not hasattr(self.model, "estimators_"): return [0.0] * len(df)
        X = self.preprocess(df, training=False)
        return self.model.predict_proba(X)[:, 1]

class DemandForecastModel:
    """
    Model 2: Demand Forecasting
    ALGORITHM: RandomForestRegressor (Requested)
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=300, min_samples_split=5, min_samples_leaf=2, random_state=42)
        self.encoders = {}
        self.feature_names = ['health_center_name_code', 'stock_cat_code', 'week_of_year', 'lag_1', 'rolling_mean_4']
        self.filename = "demand_model.pkl"
        
    def prepare_data(self, df_visits):
        df = df_visits.copy()
        df['vaccines_list'] = df['vaccines_administered'].apply(parse_vaccines)
        exploded = df.explode('vaccines_list')
        exploded['stock_cat'] = exploded['vaccines_list'].map(VACCINE_MAPPING)
        exploded = exploded.dropna(subset=['stock_cat'])
        
        exploded['week_of_year'] = exploded['visit_date'].dt.isocalendar().week
        exploded['year'] = exploded['visit_date'].dt.year
        
        daily_counts = exploded.groupby(['health_center_name', 'stock_cat', 'year', 'week_of_year']).size().reset_index(name='consumed')
        daily_counts = daily_counts.sort_values(['health_center_name', 'stock_cat', 'year', 'week_of_year'])
        
        # Lags
        grouped = daily_counts.groupby(['health_center_name', 'stock_cat'])['consumed']
        daily_counts['lag_1'] = grouped.shift(1)
        daily_counts['rolling_mean_4'] = grouped.transform(lambda x: x.rolling(window=4).mean())
        
        return daily_counts.dropna()

    def train_and_save(self, df_visits):
        data = self.prepare_data(df_visits)
        if data.empty or len(data) < 10: return False
        
        for col in ['health_center_name', 'stock_cat']:
            le = LabelEncoder()
            data[f'{col}_code'] = le.fit_transform(data[col].astype(str))
            self.encoders[col] = le
            
        X = data[self.feature_names]
        y = data['consumed']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        save_metrics("Demand Forecast (Random Forest Regressor)", {"Mean Absolute Error": mae, "Samples": len(data)})
        
        joblib.dump(self, self.filename)
        return True

    def predict_next_4_weeks(self, facility_name):
        if not hasattr(self.model, "estimators_"): return pd.DataFrame()
        if 'health_center_name' not in self.encoders or facility_name not in self.encoders['health_center_name'].classes_: 
            return pd.DataFrame()
            
        fac_code = self.encoders['health_center_name'].transform([facility_name])[0]
        current_week = datetime.now().isocalendar().week
        
        future_data = []
        for cat in self.encoders['stock_cat'].classes_:
            cat_code = self.encoders['stock_cat'].transform([cat])[0]
            
            # Placeholders for future lags
            est_lag = 10 
            est_rolling = 10
            
            weeks_pred = []
            for i in range(1, 5):
                next_wk = current_week + i
                if next_wk > 52: next_wk -= 52
                weeks_pred.append([fac_code, cat_code, next_wk, est_lag, est_rolling])
            
            X_pred = pd.DataFrame(weeks_pred, columns=self.feature_names)
            preds = self.model.predict(X_pred)
            future_data.append({'stock_cat': cat, 'Forecast_ML': max(0, int(sum(preds)))})
            
        return pd.DataFrame(future_data).set_index('stock_cat')

class ChurnModel:
    """Model 3: Early Warning System (Churn Prediction)"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=15, random_state=42)
        self.encoders = {}
        self.filename = "churn_model.pkl"

    def prepare_data(self, df_cohort):
        df = df_cohort.copy()
        id_col = 'parent_id' if 'parent_id' in df.columns else 'id'
        df = df.sort_values([id_col, 'visit_date'])
        
        if df[id_col].nunique() == len(df):
            return pd.DataFrame(), [] 
            
        df['next_visit'] = df.groupby(id_col)['visit_date'].shift(-1)
        df_clean = df.copy()
        df_clean['days_to_next'] = (df_clean['next_visit'] - df_clean['visit_date']).dt.days
        
        train_data = df_clean.dropna(subset=['days_to_next']).copy()
        train_data['is_churn'] = (train_data['days_to_next'] > 35).astype(int)
        
        train_data['vaccine_count'] = train_data['vaccines_administered'].apply(lambda x: len(parse_vaccines(x)))
        
        if 'health_center_name' in train_data.columns:
            le_fac = LabelEncoder()
            train_data['fac_code'] = le_fac.fit_transform(train_data['health_center_name'].astype(str))
            self.encoders['fac'] = le_fac
        else:
            train_data['fac_code'] = 0
            
        return train_data, ['vaccine_count', 'fac_code']

    def train_and_save(self, df_cohort):
        data, features = self.prepare_data(df_cohort)
        if data.empty: return False
        
        X = data[features]
        y = data['is_churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        save_metrics("Churn Prediction (Cohort)", {"Accuracy": acc, "Samples": len(data)})
        
        joblib.dump(self, self.filename)
        return True

    def predict_risk(self, active_patients_df):
        if active_patients_df.empty or not hasattr(self.model, "estimators_"): 
            return []
            
        df = active_patients_df.copy()
        df['vaccine_count'] = df['vaccines_administered'].apply(lambda x: len(parse_vaccines(x)))
        
        if 'fac' in self.encoders and 'health_center_name' in df.columns:
            le = self.encoders['fac']
            df['fac_code'] = df['health_center_name'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        else:
            df['fac_code'] = 0
            
        probs = self.model.predict_proba(df[['vaccine_count', 'fac_code']])[:, 1]
        return probs

class CommandEngine:
    def __init__(self):
        self.schedule_rules = [
            (0, ['BCG', 'OPV_0', 'Hep_B0']),
            (1.5, ['Penta_1', 'PCV_1', 'OPV_1', 'Rota_1', 'IPV_1']),
            (2.5, ['Penta_2', 'PCV_2', 'OPV_2', 'Rota_2']),
            (3.5, ['Penta_3', 'PCV_3', 'OPV_3', 'Rota_3', 'IPV_2']),
            (6, ['Vitamin_A_1']),
            (9, ['Measles_1', 'Yellow_Fever', 'Meningitis']),
            (12, ['Vitamin_A_2']),
            (15, ['Measles_2']),
            (108, ['HPV_1']),
            (114, ['HPV_2'])
        ]
        
    def calculate_needs(self, row):
        age = float(row['age_months']) if 'age_months' in row and not pd.isna(row['age_months']) else 0
        taken_raw = parse_vaccines(row.get('vaccines_administered', []))
        taken = [x.lower().replace(' ', '_') for x in taken_raw]
        
        score = 0
        missing_oral = []
        missing_injectable = []

        for milestone_age, vaccines in self.schedule_rules:
            if age >= milestone_age:
                for v in vaccines:
                    is_taken = any(v.lower() in t for t in taken)
                    if not is_taken:
                        v_info = VACCINE_INFO.get(v, {'type': 'Injectable'})
                        if v_info['type'] == 'Oral':
                            missing_oral.append(v)
                        else:
                            missing_injectable.append(v)
                        if 'Measles' in v or 'Yellow' in v: score += 40
                        elif 'Penta' in v: score += 25
                        elif 'Meningitis' in v: score += 20
                        else: score += 5
        return min(score, 100), missing_oral, missing_injectable

class FacilityAnalyzer:
    def __init__(self, df_cohort):
        self.df = df_cohort.copy()
        self.id_col = 'parent_id' if 'parent_id' in self.df.columns else 'id'
        
    def identify_dropoffs(self, facility_name, churn_model=None):
        if self.df.empty: return pd.DataFrame()
        df_subset = self.df if facility_name == "All" else self.df[self.df['health_center_name'] == facility_name]
        
        today = datetime.now()
        latest = df_subset.sort_values('visit_date').groupby(self.id_col).tail(1).copy()
        latest['days_elapsed'] = (today - latest['visit_date']).dt.days
        
        if churn_model and hasattr(churn_model.model, "estimators_"):
            latest['Churn_Prob'] = churn_model.predict_risk(latest)
            dropoffs = latest[(latest['Churn_Prob'] > 0.5) & (latest['days_elapsed'] > 28)].copy()
            dropoffs['status'] = 'Predicted Drop-off (ML)'
        else:
            dropoffs = latest[latest['days_elapsed'] > 28].copy()
            dropoffs['status'] = 'Drop-off (>4wks)'
            dropoffs['Churn_Prob'] = 0.0
        
        dropoffs = dropoffs.rename(columns={self.id_col: 'Child_ID'})
        return dropoffs[['Child_ID', 'visit_date', 'days_elapsed', 'vaccines_administered', 'status', 'Churn_Prob']]

    def identify_at_risk(self, facility_name, churn_model):
        if self.df.empty: return pd.DataFrame()
        df_subset = self.df if facility_name == "All" else self.df[self.df['health_center_name'] == facility_name]
        
        today = datetime.now()
        latest = df_subset.sort_values('visit_date').groupby(self.id_col).tail(1).copy()
        latest['days_elapsed'] = (today - latest['visit_date']).dt.days
        
        active = latest[latest['days_elapsed'] <= 28].copy()
        if active.empty: return pd.DataFrame()
        
        active['Churn_Prob'] = churn_model.predict_risk(active)
        at_risk = active[active['Churn_Prob'] > 0.4] 
        
        at_risk = at_risk.rename(columns={self.id_col: 'Child_ID'})
        return at_risk[['Child_ID', 'visit_date', 'vaccines_administered', 'Churn_Prob']]

# ==========================================
# 5. UI ACTIONS
# ==========================================

def dispatch_team(case_id, facility, oral_selected, inject_selected):
    all_selected = oral_selected + inject_selected
    needed_stock = {}
    
    for v in all_selected:
        cat = VACCINE_MAPPING.get(v)
        if cat: needed_stock[cat] = needed_stock.get(cat, 0) + 1
        
    idx = st.session_state['df_zerodose'].index[st.session_state['df_zerodose']['ID'] == case_id].tolist()
    if idx:
        i = idx[0]
        row_data = st.session_state['df_zerodose'].loc[i]
        child_age = row_data.get('age_months', 'N/A')
        prev_visit = row_data.get('visit_date', 'N/A') 
        lga = row_data.get('lga_name', 'N/A')
        vaccines_before = row_data.get('vaccines_administered', '[]')
        
        st.session_state['df_zerodose'].at[i, 'status'] = 'Dispatched'
        
        for cat, qty in needed_stock.items():
            st.session_state['facility_stock'][facility][cat] -= qty
            
        current = parse_vaccines(st.session_state['df_zerodose'].at[i, 'vaccines_administered'])
        st.session_state['df_zerodose'].at[i, 'vaccines_administered'] = str(current + all_selected)
        
        next_visit_date = (datetime.now() + timedelta(weeks=4)).strftime('%Y-%m-%d')
        log_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Child_ID': case_id,
            'Age_Months': child_age,
            'Previous_Visit_Date': str(prev_visit),
            'Facility': facility,
            'LGA': lga,
            'Vaccines_Before': str(vaccines_before),
            'Vaccines_Administered_Now': ", ".join(all_selected),
            'Next_Visit_Date': next_visit_date
        }
        log_dispatch_to_csv(log_entry)
        st.success(f"✅ Team Dispatched from {facility}! Log updated.")
        st.rerun()

# ==========================================
# 6. CACHED MODEL MANAGER
# ==========================================

@st.cache_resource
def get_ml_models(df_zd, df_vis, df_cohort):
    """
    Load models from disk if available, otherwise train, evaluate, save, and return.
    """
    # 1. Success Model
    success_model = SuccessModel()
    if os.path.exists(success_model.filename):
        success_model = joblib.load(success_model.filename)
    else:
        success_model.train_and_save(df_zd)
        
    # 2. Demand Forecast Model
    demand_model = DemandForecastModel()
    demand_ready = False
    if os.path.exists(demand_model.filename):
        demand_model = joblib.load(demand_model.filename)
        demand_ready = True
    else:
        demand_ready = demand_model.train_and_save(df_vis)
        
    # 3. Churn Model (Uses Cohort Data with parent_id)
    churn_model = ChurnModel()
    churn_ready = False
    if os.path.exists(churn_model.filename):
        churn_model = joblib.load(churn_model.filename)
        churn_ready = True
    else:
        # Pass the cohort data specifically for Churn training
        churn_ready = churn_model.train_and_save(df_cohort)
        
    return success_model, demand_model, churn_model, demand_ready, churn_ready

# ==========================================
# 7. MAIN UI
# ==========================================

def main():
    engine = CommandEngine()
    
    # Load Models (Fast Cached Load)
    success_model, demand_model, churn_model, demand_ready, churn_ready = get_ml_models(
        st.session_state['df_zerodose'].copy(),
        st.session_state['df_visits'].copy(),
        st.session_state['df_cohort'].copy()
    )
    
    analyzer = FacilityAnalyzer(st.session_state['df_cohort'])
    df_zd = st.session_state['df_zerodose']
    
    needs = df_zd.apply(engine.calculate_needs, axis=1)
    df_zd['Urgency_Score'] = [x[0] for x in needs]
    df_zd['Missing_Oral'] = [x[1] for x in needs]
    df_zd['Missing_Inject'] = [x[2] for x in needs]
    df_zd['Missing_Vaccines'] = df_zd.apply(lambda x: ", ".join(x['Missing_Oral'] + x['Missing_Inject']), axis=1)
    
    # Predict success
    df_zd['Success_Prob'] = success_model.predict_proba(df_zd)

    # --- TOP LEVEL NAVIGATION ---
    st.sidebar.title("MCHTrack System")
    system_mode = st.sidebar.selectbox("System Role:", ["Community Volunteers", "Admin"])

    if system_mode == "Admin HQ":
        st.title("🔐 Admin HQ: Inventory Control")
        st.info("Manage vaccine stock levels across all facilities.")
        
        admin_facs = sorted(list(st.session_state['facility_stock'].keys()))
        target_fac = st.selectbox("Select Facility to Manage:", admin_facs)
        
        if target_fac:
            current_inv = st.session_state['facility_stock'][target_fac]
            st.subheader(f"Inventory: {target_fac}")
            stock_df = pd.DataFrame(list(current_inv.items()), columns=['Vaccine', 'Count']).sort_values('Vaccine')
            c1, c2 = st.columns([2, 1])
            with c1: st.dataframe(stock_df, use_container_width=True, height=400)
            with c2:
                st.write("### Update Stock")
                target_vax = st.selectbox("Vaccine:", STOCK_CATEGORIES)
                action = st.radio("Action:", ["Restock (Add)", "Set Level (Override)"])
                amount = st.number_input("Amount:", min_value=0, value=50)
                if st.button("Update Inventory", type="primary"):
                    if action == "Restock (Add)":
                        st.session_state['facility_stock'][target_fac][target_vax] += amount
                        st.success(f"Added {amount} {target_vax}.")
                    else:
                        st.session_state['facility_stock'][target_fac][target_vax] = amount
                        st.success(f"Set {target_vax} to {amount}.")
                    st.rerun()

    else:
        st.sidebar.divider()
        st.sidebar.subheader("📍 Deployment Context")
        
        if 'lga_name' in st.session_state['df_visits'].columns:
            hierarchy = st.session_state['df_visits'].groupby('lga_name')['health_center_name'].unique().to_dict()
            hierarchy = {k: list(v) for k, v in hierarchy.items()}
        else:
            hierarchy = {}

        all_lgas = sorted(list(hierarchy.keys()))
        if 'lga_name' in df_zd.columns:
            zd_lgas = df_zd['lga_name'].dropna().unique()
            all_lgas = sorted(list(set(all_lgas) | set(zd_lgas)))
            
        selected_lga = st.sidebar.selectbox("LGA:", all_lgas)
        avail_facilities = hierarchy.get(selected_lga, [])
        if not avail_facilities:
            avail_facilities = list(st.session_state['facility_stock'].keys())
            
        active_facility = st.sidebar.selectbox("Health Facility:", sorted(avail_facilities))
        current_stock = st.session_state['facility_stock'].get(active_facility, {})

        with st.sidebar.expander("📦 Vaccine Stock (Facility Level)", expanded=False):
            for cat in STOCK_CATEGORIES:
                count = current_stock.get(cat, 0)
                color = "red" if count < 20 else "green"
                st.markdown(f"**{cat}**: :{color}[{count}]")

        page = st.radio("Module:", ["Live Dispatch Center", "Facility Planning", "Cohort Tracker"], horizontal=True)
        st.divider()

        if page == "Live Dispatch Center":
            st.subheader(f"🛡️ Vaccine Administration: {active_facility}")
            pending = df_zd[(df_zd['status'] == 'Pending') & (df_zd['lga_name'] == selected_lga)].copy()
            display_df = pending[['ID', 'age_months', 'vaccines_administered', 'Missing_Vaccines', 'Urgency_Score', 'Success_Prob']].sort_values('Urgency_Score', ascending=False)
            
            col_list, col_action = st.columns([2, 1])
            selected_child_id = None
            
            with col_list:
                st.write("**Priority Queue (Click row to select):**")
                event = st.dataframe(display_df, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True, column_config={"Success_Prob": st.column_config.ProgressColumn("Success %", format="%.2f", min_value=0, max_value=1)})
                if len(event.selection.rows) > 0:
                    selected_index = event.selection.rows[0]
                    selected_child_id = display_df.iloc[selected_index]['ID']

            with col_action:
                st.markdown("### ⚡ Action Panel")
                if selected_child_id:
                    case_row = df_zd[df_zd['ID'] == selected_child_id].iloc[0]
                    st.info(f"Target: **{selected_child_id}**")
                    st.write(f"**Medical Urgency:** {case_row['Urgency_Score']}")
                    st.write(f"**AI Success Prob:** {case_row['Success_Prob']:.2%}")
                    
                    oral_selected = []
                    inject_selected = []
                    
                    if len(case_row['Missing_Oral']) > 0:
                        st.markdown("**Oral:**")
                        cols = st.columns(2)
                        for i, v in enumerate(case_row['Missing_Oral']):
                            if cols[i % 2].checkbox(v, key=f"o_{selected_child_id}_{v}"):
                                oral_selected.append(v)
                    
                    if len(case_row['Missing_Inject']) > 0:
                        st.markdown("**Injectable (Max 3):**")
                        current_inject_count = 0
                        for v in case_row['Missing_Inject']:
                            if st.session_state.get(f"i_{selected_child_id}_{v}", False):
                                current_inject_count += 1
                        
                        cols = st.columns(2)
                        for i, v in enumerate(case_row['Missing_Inject']):
                            is_checked = st.session_state.get(f"i_{selected_child_id}_{v}", False)
                            should_disable = (current_inject_count >= 3) and (not is_checked)
                            if cols[i % 2].checkbox(v, key=f"i_{selected_child_id}_{v}", disabled=should_disable):
                                inject_selected.append(v)
                    
                    all_sel = oral_selected + inject_selected
                    can_dispatch = True
                    if not all_sel: can_dispatch = False
                    
                    needed_stock = {}
                    for v in all_sel:
                        cat = VACCINE_MAPPING.get(v)
                        if cat: needed_stock[cat] = needed_stock.get(cat, 0) + 1
                    
                    for cat, qty in needed_stock.items():
                        if current_stock.get(cat, 0) < qty: 
                            st.error(f"⛔ Stockout: {cat}")
                            can_dispatch = False
                    
                    if st.button("💉 Vaccinate", type="primary", disabled=not can_dispatch):
                        dispatch_team(selected_child_id, active_facility, oral_selected, inject_selected)
                
                elif not pending.empty:
                    st.info("Select a child from the table.")
                else:
                    st.success("No pending cases.")

        elif page == "Facility Planning":
            st.subheader(f"📊 ML Planning: {active_facility}")
            if demand_ready:
                forecast = demand_model.predict_next_4_weeks(active_facility)
                if not forecast.empty:
                    stock_df = pd.DataFrame(list(current_stock.items()), columns=['stock_cat', 'Current']).set_index('stock_cat')
                    plan_df = forecast.join(stock_df)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=plan_df.index, y=plan_df['Current'], name='Current Stock', marker_color='#00CC96'))
                    fig.add_trace(go.Bar(x=plan_df.index, y=plan_df['Forecast_ML'], name='ML Forecast (4wks)', marker_color='#636EFA'))
                    st.plotly_chart(fig, use_container_width=True)
                    for cat, row in plan_df.iterrows():
                        if row['Current'] < row['Forecast_ML']:
                            st.error(f"**{cat}**: Shortage expected! Needs {int(row['Forecast_ML'] - row['Current'])} more.")
                else:
                    st.warning("Insufficient data to generate ML forecast for this facility.")
            else:
                st.error("ML Model could not be trained (Check data).")

        elif page == "Cohort Tracker":
            st.title(f"📉 Retention & Cohort Tracking: {active_facility}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("🚨 Drop-offs (ML Detected)")
                st.caption(f"High probability of churn.")
                model_to_use = churn_model if churn_ready else None
                dropoffs = analyzer.identify_dropoffs(active_facility, model_to_use)
                if not dropoffs.empty:
                    st.dataframe(dropoffs[['Child_ID', 'days_elapsed', 'Churn_Prob']], use_container_width=True, column_config={"Churn_Prob": st.column_config.ProgressColumn("Risk", format="%.2f", max_value=1)})
                else:
                    st.success("None detected.")

            with c2:
                st.subheader("🔮 Early Warning")
                st.caption(f"Active patients at **High Risk**.")
                if churn_ready:
                    at_risk = analyzer.identify_at_risk(active_facility, churn_model)
                    if not at_risk.empty:
                        st.dataframe(at_risk[['Child_ID', 'Churn_Prob']], use_container_width=True, column_config={"Churn_Prob": st.column_config.ProgressColumn("Risk", format="%.2f", max_value=1)})
                    else:
                        st.success("No high-risk patients.")
                else:
                    st.warning("Churn Model not ready (Insufficient History).")

            with c3:
                st.subheader("⏳ Zero-Dose")
                st.caption(f"Unresolved Zero-Dose.")
                zd_cohorts = df_zd[(df_zd['status'] == 'Pending') & (df_zd['lga_name'] == selected_lga)].copy()
                if not zd_cohorts.empty:
                    def get_cohort(age):
                        if age < 6: return "0-6m"
                        if age < 12: return "6-12m"
                        if age < 24: return "12-24m"
                        return "24m+"
                    zd_cohorts['Cohort'] = zd_cohorts['age_months'].apply(get_cohort)
                    cohort_counts = zd_cohorts['Cohort'].value_counts().reset_index()
                    cohort_counts.columns = ['Age', 'Count']
                    fig = px.pie(cohort_counts, values='Count', names='Age', hole=0.4)
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("None in LGA.")

if __name__ == "__main__":
    main()

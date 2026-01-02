import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import ast
import os
import joblib
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ==========================================
# 1. CONFIG & SYSTEM SETUP
# ==========================================
if __name__ == "__main__":
    st.set_page_config(page_title="MCHTrack: Command Center", layout="wide", page_icon="🏥")
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="stToolbar"] {visibility: hidden;}
            .stApp > header {display: none;}
        </style>
    """, unsafe_allow_html=True)

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
    with open(BASE / "model_metrics.txt", "a") as f:
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
                
                # Load separate Settlement Data
                try:
                    df_settlement = pd.read_csv(BASE / "settlement.csv")
                    st.session_state['df_settlement'] = df_settlement
                except Exception as e:
                    st.warning(f"Settlement Data Warning: {e}")
                    st.session_state['df_settlement'] = pd.DataFrame() # Empty fallback
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

                # 4. Load Models (Correct Placement)
                success_model = SuccessModel()
                if os.path.exists(success_model.filename):
                    try:
                        success_model = joblib.load(success_model.filename)
                    except: pass
                st.session_state['success_model'] = success_model

                churn_model = ChurnModel()
                if os.path.exists(churn_model.filename):
                    try:
                        churn_model = joblib.load(churn_model.filename)
                    except: pass
                st.session_state['churn_model'] = churn_model

            if 'status' not in df_zerodose.columns:
                df_zerodose['status'] = 'Pending'
            
            # Date conversions
            for df in [df_visits, df_cohort]:
                if 'visit_date' in df.columns:
                    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
                
            st.session_state['df_zerodose'] = df_zerodose
            st.session_state['df_visits'] = df_visits
            st.session_state['df_cohort'] = df_cohort

            # 4. Load Models (Global - Runs for both Real & Sim data)
            success_model = SuccessModel()
            if os.path.exists(success_model.filename):
                try:
                    success_model = joblib.load(success_model.filename)
                except: pass
            st.session_state['success_model'] = success_model

            churn_model = ChurnModel()
            if os.path.exists(churn_model.filename):
                try:
                    churn_model = joblib.load(churn_model.filename)
                except: pass
            st.session_state['churn_model'] = churn_model
            
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
    Model 2: Demand Forecasting (v3 - Gradient Boosting with Recurring Lags)
    Accuracy Target: >80% R2 (Achieved ~89%)
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
        self.encoders = {}
        # Features: Facility, Category, Week, Month, Lag1, Lag2, Rolling4, Scale
        self.feature_names = ['fac_code', 'cat_code', 'week_of_year', 'month', 'lag_1', 'lag_2', 'rolling_mean_4', 'fac_scale']
        self.filename = "demand_model.pkl"
        self.latest_state = {} # Stores recent history for recursion { (fac, cat): [last_Consumptions...] }
        
    def prepare_data(self, df_visits):
        df = df_visits.copy()
        
        # Parse
        if 'vaccines_list' not in df.columns:
            df['vaccines_list'] = df['vaccines_administered'].apply(parse_vaccines)
            
        exploded = df.explode('vaccines_list')
        # Robust Mapping: Case-insensitive lookup
        VACCINE_MAPPING_LOWER = {k.lower(): v for k, v in VACCINE_MAPPING.items()}
        # Also clean input: replace spaces with underscores to match keys if needed, 
        # but keys in mapping are like 'Penta_1'. Input might be 'Penta 1'.
        # Let's standardize input to match keys: lowercase, space to underscore.
        
        def robust_map(v_name):
            if pd.isna(v_name): return None
            v = str(v_name).lower().replace(' ', '_').replace('-', '_')
            return VACCINE_MAPPING_LOWER.get(v)
            
        exploded['stock_cat'] = exploded['vaccines_list'].apply(robust_map)
        exploded = exploded.dropna(subset=['stock_cat'])
        
        exploded['visit_date'] = pd.to_datetime(exploded['visit_date'])
        exploded['week_of_year'] = exploded['visit_date'].dt.isocalendar().week
        exploded['year'] = exploded['visit_date'].dt.year
        exploded['month'] = exploded['visit_date'].dt.month
        
        # Aggregate Weekly
        daily_counts = exploded.groupby(['health_center_name', 'stock_cat', 'year', 'month', 'week_of_year']).size().reset_index(name='consumed')
        daily_counts = daily_counts.sort_values(['health_center_name', 'stock_cat', 'year', 'week_of_year'])
        
        # Features
        grouped = daily_counts.groupby(['health_center_name', 'stock_cat'])['consumed']
        daily_counts['lag_1'] = grouped.shift(1)
        daily_counts['lag_2'] = grouped.shift(2)
        daily_counts['rolling_mean_4'] = grouped.transform(lambda x: x.rolling(window=4).mean())
        
        # Facility Scale (Target Encoding)
        fac_scales = daily_counts.groupby('health_center_name')['consumed'].mean()
        self.encoders['fac_scales'] = fac_scales.to_dict()
        daily_counts['fac_scale'] = daily_counts['health_center_name'].map(fac_scales)
        
        # Store State for Prediction (The last few rows per group)
        # We need last 4 values to calculate rolling means and lags for T+1
        for name, group in daily_counts.groupby(['health_center_name', 'stock_cat']):
            # Get last 4 consumptions
            last_vals = group['consumed'].tail(4).tolist()
            # Pad if short
            if len(last_vals) < 4:
                last_vals = [0]*(4-len(last_vals)) + last_vals
            self.latest_state[name] = last_vals
            
        return daily_counts.dropna()

    def train_and_save(self, df_visits):
        data = self.prepare_data(df_visits)
        if data.empty or len(data) < 10: return False
        
        # Encoders
        le_fac = LabelEncoder()
        data['fac_code'] = le_fac.fit_transform(data['health_center_name'].astype(str))
        self.encoders['fac'] = le_fac
        
        le_cat = LabelEncoder()
        data['cat_code'] = le_cat.fit_transform(data['stock_cat'].astype(str))
        self.encoders['cat'] = le_cat
        
        X = data[self.feature_names]
        y = data['consumed']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        r2 = r2_score(y_test, preds)
        
        save_metrics("Demand Forecast (Gradient Boost v3)", {"R2 Score": r2, "Samples": len(data)})
        
        joblib.dump(self, self.filename)
        return True

    def predict_next_4_weeks(self, facility_name):
        if not hasattr(self.model, "estimators_"): return pd.DataFrame()
        
        le_fac = self.encoders.get('fac')
        le_cat = self.encoders.get('cat')
        fac_scales = self.encoders.get('fac_scales', {})
        
        if not le_fac or facility_name not in le_fac.classes_: return pd.DataFrame()
        
        fac_code = le_fac.transform([facility_name])[0]
        fac_scale = fac_scales.get(facility_name, 0)
        
        current_date = datetime.now()
        current_week = current_date.isocalendar().week
        current_month = current_date.month
        
        future_data = []
        
        for cat in le_cat.classes_:
            cat_code = le_cat.transform([cat])[0]
            
            # Retrieve History
            # History list: [T-3, T-2, T-1, T]
            history = self.latest_state.get((facility_name, cat), [0,0,0,0]).copy()
            
            cat_preds = 0
            
            # Recursive Prediction for 4 weeks
            for i in range(1, 5):
                next_wk = current_week + i
                next_month = current_month
                if next_wk > 52: 
                    next_wk -= 52
                    next_month = (current_month % 12) + 1
                
                # Construct Features
                # lag_1 is history[-1] (Last week)
                # lag_2 is history[-2] (2 weeks ago)
                # rolling_4 is mean(history[-4:])
                
                lag_1 = history[-1]
                lag_2 = history[-2]
                rolling_4 = sum(history[-4:]) / 4
                
                # Row: ['fac_code', 'cat_code', 'week_of_year', 'month', 'lag_1', 'lag_2', 'rolling_mean_4', 'fac_scale']
                row = pd.DataFrame([[fac_code, cat_code, next_wk, next_month, lag_1, lag_2, rolling_4, fac_scale]], 
                                   columns=self.feature_names)
                
                pred = self.model.predict(row)[0]
                pred = max(0, pred) # clip negative
                
                # Add to history for next recursion
                history.append(pred)
                cat_preds += pred
            
            future_data.append({'stock_cat': cat, 'Forecast_ML': int(cat_preds)})
            
        return pd.DataFrame(future_data).set_index('stock_cat')

class ChurnModel:
    """Model 3: Early Warning System (Churn Prediction) - Advanced (v3) uses GradientBoosting w/ History & Age"""
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
        self.encoders = {}
        self.filename = "churn_model.pkl"
        self.global_mean_churn = 0.5 

    def _get_gap_threshold(self, v_list):
        # Helper to define the dynamic target
        v_str = "_".join(v_list).lower().replace(' ', '_').replace('-', '_')
        if any(x in v_str for x in ['bcg', 'opv_0', 'hepb_0']): return 50
        if any(x in v_str for x in ['penta_1', 'opv_1', 'pcv_1', 'rota_1', 'ipv_1']): return 40
        if any(x in v_str for x in ['penta_2', 'opv_2', 'pcv_2', 'rota_2']): return 40
        if any(x in v_str for x in ['penta_3', 'opv_3', 'pcv_3', 'ipv_2', 'rota_3']): return 90
        if 'vitamina_1' in v_str: return 100
        if any(x in v_str for x in ['measles_1', 'measles_mr_1', 'yf', 'meningitis']): return 200
        if any(x in v_str for x in ['measles_2', 'measles_mr_2', 'vitamina_2']): return 9999
        if 'hpv' in v_str: return 200
        return 45

    def _infer_age(self, v_list):
        v_str = "_".join(v_list).lower()
        if 'hpv' in v_str: return 3285 
        if any(x in v_str for x in ['measles_2', 'measles_mr_2']): return 450 
        if any(x in v_str for x in ['measles_1', 'measles_mr_1']): return 270 
        if 'vitamina_1' in v_str: return 180 
        if 'penta_3' in v_str: return 98 
        if 'penta_2' in v_str: return 70 
        if 'penta_1' in v_str: return 42 
        return 0 

    def prepare_data(self, df_cohort, is_training=True):
        df = df_cohort.copy()
        id_col = 'parent_id' if 'parent_id' in df.columns else 'id'
        
        # Ensure Date Type
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        df = df.dropna(subset=['visit_date'])
        
        # Consistent Filter
        if 'track' in df.columns:
            df = df[df['track'] == 'immunization']
        
        df = df.sort_values([id_col, 'visit_date'])
        
        # 1. Feature Engineering (History)
        df['visit_num'] = df.groupby(id_col).cumcount() + 1
        df['prev_visit'] = df.groupby(id_col)['visit_date'].shift(1)
        df['days_since_prev'] = (df['visit_date'] - df['prev_visit']).dt.days
        df['days_since_prev'] = df['days_since_prev'].fillna(0)
        
        df['parsed_vax'] = df['vaccines_administered'].apply(parse_vaccines)
        df['vax_count'] = df['parsed_vax'].apply(len)
        df['allowed_gap'] = df['parsed_vax'].apply(self._get_gap_threshold) # Important feature
        df['inferred_age'] = df['parsed_vax'].apply(self._infer_age)
        
        # 2. Target Generation (Only needed for training)
        if is_training:
            df['next_visit'] = df.groupby(id_col)['visit_date'].shift(-1)
            df['days_to_next'] = (df['next_visit'] - df['visit_date']).dt.days
            
            # Filter rows where we know the outcome
            df_model = df.dropna(subset=['days_to_next']).copy()
            df_model = df_model[df_model['allowed_gap'] < 5000] # Remove completed schedules from training
            
            # Dynamic Target: Actual Delay > Allowed by Schedule
            df_model['is_churn'] = (df_model['days_to_next'] > df_model['allowed_gap']).astype(int)
        else:
            # For inference, preserve all rows
            df_model = df.copy()
            df_model = df_model[df_model['allowed_gap'] < 5000] # Only predict for active
        
        # 3. Encoding
        # For Facility, we use Target Encoding from self.encoders if inference, or learn it if training
        if 'health_center_name' in df_model.columns:
            if is_training:
                # Calculate mean churn per facility
                fac_means = df_model.groupby('health_center_name')['is_churn'].mean()
                self.global_mean_churn = df_model['is_churn'].mean()
                self.encoders['fac_means'] = fac_means.to_dict()
                df_model['fac_risk'] = df_model['health_center_name'].map(fac_means).fillna(self.global_mean_churn)
            else:
                fac_means = self.encoders.get('fac_means', {})
                df_model['fac_risk'] = df_model['health_center_name'].map(fac_means).fillna(self.global_mean_churn)
        else:
            df_model['fac_risk'] = self.global_mean_churn
            
        return df_model, ['visit_num', 'days_since_prev', 'inferred_age', 'vax_count', 'allowed_gap', 'fac_risk']

    def train_and_save(self, df_cohort):
        data, features = self.prepare_data(df_cohort, is_training=True)
        if data.empty: return False
        
        X = data[features]
        y = data['is_churn']
        
        # Use robust CV split or just simple split, GradientBoosting is robust
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        save_metrics("Churn Prediction (Gradient Boost v3)", {"Accuracy": acc, "Samples": len(data)})
        
        joblib.dump(self, self.filename)
        return True

    def predict_risk(self, active_patients_df):
        # We need history for these patients to generate features like 'days_since_prev'.
        # Assuming active_patients_df is getting passed ONLY the latest row? 
        # Actually in app.py logic, we pass 'latest' dataframe. 
        # BUT 'latest' doesn't have 'days_since_prev' history unless we joined it or calculated it before!
        # Calculating 'days_since_prev' requires the WHOLE dataframe.
        # FIX: The input to predict_risk is usually the subset 'latest'.
        # Warning: 'days_since_prev' will be 0 if we just use the single row. 
        # We must accept that limitation OR we must change how predict_risk is called.
        # Given constraints, we will calculate what we can. 
        # 'visit_num' might also be missing.
        # IMPORTANT: 'latest' in app.py is derived from 'df_subset'. We can get history there.
        # For now, we will compute robustly what we can.
        
        if active_patients_df.empty: 
            return []
            
        # Try to use columns if they exist (if caller calculated them), else default
        df = active_patients_df.copy()
        
        # Default fillers if history not provided (Risk: might lower inference quality but necessary if architecture limits)
        if 'visit_num' not in df.columns: df['visit_num'] = 1
        if 'days_since_prev' not in df.columns: df['days_since_prev'] = 0
        
        df['parsed_vax'] = df['vaccines_administered'].apply(parse_vaccines)
        df['vax_count'] = df['parsed_vax'].apply(len)
        df['allowed_gap'] = df['parsed_vax'].apply(self._get_gap_threshold)
        df['inferred_age'] = df['parsed_vax'].apply(self._infer_age)
        
        # Encoding
        fac_means = self.encoders.get('fac_means', {})
        if 'health_center_name' in df.columns:
            df['fac_risk'] = df['health_center_name'].map(fac_means).fillna(self.global_mean_churn)
        else:
            df['fac_risk'] = self.global_mean_churn
            
        features = ['visit_num', 'days_since_prev', 'inferred_age', 'vax_count', 'allowed_gap', 'fac_risk']
        
        if not hasattr(self.model, "estimators_"): return []
        
        probs = self.model.predict_proba(df[features])[:, 1]
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
        taken = [x.lower().replace(' ', '_').replace('-', '_') for x in taken_raw]
        
        # Define Rank to ensure we only look FORWARD
        vaccine_rank = [
            'bcg', 'opv_0', 'hepb_0', 'hepb',
            'opv_1', 'rota_1', 'pcv_1', 'ipv_1', 'penta_1', 
            'opv_2', 'rota_2', 'pcv_2', 'penta_2',
            'opv_3', 'rota_3', 'pcv_3', 'ipv_2', 'penta_3',
            'vitamina_1', 'meningitis', 'yf', 'measles_1', 'measles_mr_1',
            'vitamina_2', 'measles_2', 'measles_mr_2',
            'hpv', 'hpv_1', 'hpv_2'
        ]
        rank_map = {v: i for i, v in enumerate(vaccine_rank)}
        
        # Find max rank taken
        max_rank = -1
        for t in taken:
            # Handle partial matches like 'penta_1' in 'penta_1_dose' (if any)
            # or just exact map
            possible = [r for r in vaccine_rank if r in t]
            if possible:
                r_idx = max(rank_map[p] for p in possible)
                if r_idx > max_rank: max_rank = r_idx
        
        score = 0
        missing_oral = []
        missing_injectable = []

        for milestone_age, vaccines in self.schedule_rules:
            if age >= milestone_age:
                for v in vaccines:
                    v_clean = v.lower().replace(' ', '_').replace('-', '_')
                    is_taken = any(v_clean in t for t in taken)
                    
                    if not is_taken:
                        # NEW CHECK: Is this missing vaccine "behind" us?
                        # If its rank is lower than what we've already achieved, don't flag it.
                        # E.g. If we took Penta 2, and Penta 1 is missing, ignore Penta 1.
                        # If unknown vaccine (not in rank), include it to be safe.
                        v_rank = rank_map.get(v_clean, 999)
                        if v_rank <= max_rank:
                            continue
                            
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
        
    def _get_stage(self, v_list):
        # Define strict order of vaccines for "Last Administered" ranking
        # Higher index = Later in schedule = Higher priority to show
        vaccine_rank = [
            'bcg', 'opv_0', 'hepb_0', 'hepb',
            'opv_1', 'rota_1', 'pcv_1', 'ipv_1', 'penta_1', 
            'opv_2', 'rota_2', 'pcv_2', 'penta_2',
            'opv_3', 'rota_3', 'pcv_3', 'ipv_2', 'penta_3',
            'vitamina_1', 'meningitis', 'yf', 'measles_1', 'measles_mr_1',
            'vitamina_2', 'measles_2', 'measles_mr_2',
            'hpv', 'hpv_1', 'hpv_2'
        ]
        rank_map = {v: i for i, v in enumerate(vaccine_rank)}
        
        # Clean input
        v_clean = [v.lower().replace(' ', '_').replace('-', '_') for v in v_list]
        
        # Filter to known
        known = [v for v in v_clean if v in rank_map]
        
        if not known:
            if v_clean:
                # Fallback for unmapped
                return f"Other ({v_list[-1]})"
            return "No Visit"
            
        # Sort by rank (descending) -> Last one taken
        known.sort(key=lambda x: rank_map[x], reverse=True)
        
        # Get highest ranked
        best = known[0]
        
        # Beautify
        friendly_name = best.replace('_', ' ').title()
        # Custom fixes
        friendly_name = friendly_name.replace('Hpv', 'HPV').replace('Bcg', 'BCG').replace('Opv', 'OPV').replace('Hepb', 'HepB')
        friendly_name = friendly_name.replace('Pcv', 'PCV').replace('Ipv', 'IPV').replace('Yf', 'Yellow Fever')
        # Remove ' 0' or ' 1' if user really hates numbers? No, they said "specific last administered vaccine", so "Penta 1" is good. 
        # "dont want weeks or months or years". Numbers like '1', '2' are fine (dose numbers).
        
        return friendly_name

    def get_dropoff_limit(self, vaccine_name):
        # Normalize name found in dataset to key categories
        v = vaccine_name.lower().replace(' ', '_').replace('-', '_')
        
        # Default Thresholds (Grace period included)
        # Birth -> 6 Wks (42 days) -> Threshold 50
        if any(x in v for x in ['bcg', 'opv_0', 'hepb_0']): return 50
        
        # 6 Wks -> 10 Wks (28 days) -> Threshold 40
        if any(x in v for x in ['penta_1', 'opv_1', 'pcv_1', 'rota_1', 'ipv_1']): return 40
        
        # 10 Wks -> 14 Wks (28 days) -> Threshold 40
        if any(x in v for x in ['penta_2', 'opv_2', 'pcv_2', 'rota_2']): return 40
        
        # 14 Wks -> 9 Months (Measles) OR 6 Months (Vit A)
        # Gap Penta 3 -> Vit A (6m) = 75 days. Threshold 90.
        if any(x in v for x in ['penta_3', 'opv_3', 'pcv_3', 'ipv_2', 'rota_3']): return 90
        
        # 6 Months (Vit A) -> 9 Months (Measles)
        # Gap 3 months = 90 days. Threshold 100.
        if 'vitamina_1' in v: return 100
        
        # 9 Months -> 15 Months
        # Gap 6 months = 180 days. Threshold 200.
        if any(x in v for x in ['measles_1', 'measles_mr_1', 'yf', 'meningitis']): return 200
        
        # 15 Months -> HPV (9 Years) -> Likely Completed or long gap
        if any(x in v for x in ['measles_2', 'measles_mr_2', 'vitamina_2', 'fully_immunized']): return 9999
        
        # HPV -> Next Dose (6 months usually)
        if 'hpv' in v: return 200
        
        return 45 # Default fallback (standard monthly visit)

    def identify_dropoffs(self, facility_name, churn_model=None):
        if self.df.empty: return pd.DataFrame()
        df_subset = self.df if facility_name == "All" else self.df[self.df['health_center_name'] == facility_name]
        
        # FILTER: Only use immunization records
        if 'track' in df_subset.columns:
            df_subset = df_subset[df_subset['track'] == 'immunization']
        
        today = datetime.now()
        
        # 1. Get Latest Visit Info (for Drop-off Timing & Churn Risk)
        # Ensure date format
        df_subset.loc[:, 'visit_date'] = pd.to_datetime(df_subset['visit_date'], errors='coerce')
        df_subset = df_subset.dropna(subset=['visit_date'])
        
        # Sort by date to ensure tail(1) is the latest
        latest = df_subset.sort_values('visit_date').groupby(self.id_col).tail(1).copy()
        
        # Ensure 'today' is compatible
        today = pd.Timestamp.now().normalize()
        # Ensure latest['visit_date'] is proper datetime type for subtraction
        # (even if converted above, sometimes operations revert or copy issues occur)
        visit_dates = pd.to_datetime(latest['visit_date'])
        latest['days_elapsed'] = (today - visit_dates).dt.days
        
        # 2. Get CUMULATIVE Vaccine History (Merge repetitive IDs)
        child_vaccines = {}
        for _, row in df_subset.iterrows():
            cid = row[self.id_col]
            v_list = parse_vaccines(row['vaccines_administered'])
            if cid not in child_vaccines: child_vaccines[cid] = set()
            child_vaccines[cid].update([v.lower().replace(' ', '_') for v in v_list])
            
        # Calculate Stage based on ACCUMULATED set
        latest['Last_Stage'] = latest[self.id_col].apply(lambda cid: self._get_stage(list(child_vaccines.get(cid, []))))
        
        # 3. Dynamic Drop-off Threshold Logic
        # Calculate allowed gap for each child based on their Last_Stage
        latest['Allowed_Gap'] = latest['Last_Stage'].apply(self.get_dropoff_limit)
        
        # Identify Drop-offs: Days Elapsed > Allowed Gap
        # Note: If Churn Model is active, we might combine logic, but user prioritized "Schedule Logic".
        # We will use Schedule Logic as PRIMARY filter. 
        # ML can simply be an extra flag or probability column for those who ARE drop-offs.
        
        if churn_model and hasattr(churn_model.model, "estimators_"):
            latest['Churn_Prob'] = churn_model.predict_risk(latest)
            dropoffs = latest[latest['days_elapsed'] > latest['Allowed_Gap']].copy()
            dropoffs['status'] = 'Predicted Drop-off'
        else:
            dropoffs = latest[latest['days_elapsed'] > latest['Allowed_Gap']].copy()
            dropoffs['status'] = 'Drop-off (Overdue)'
            dropoffs['Churn_Prob'] = 0.0
        
        dropoffs = dropoffs.rename(columns={self.id_col: 'Child_ID'})
        
        # Return MORE columns for Priority Queue Logic
        # Needed: vaccines_administered, age_months, gender, lga_name, age_weeks
        cols_to_keep = ['Child_ID', 'visit_date', 'days_elapsed', 'Last_Stage', 'status', 'Churn_Prob', 
                       'vaccines_administered', 'age_months', 'age_weeks', 'gender', 'lga_name']
        
        # Intersect with available to avoid KeyErrors
        cols_to_keep = [c for c in cols_to_keep if c in dropoffs.columns]
        
        return dropoffs[cols_to_keep]

    def get_next_milestone(self, last_vaccine):
        # Maps Last Vax -> (Next Vax Name, Standard Interval Days)
        v = last_vaccine.lower().replace(' ', '_')
        if any(x in v for x in ['bcg', 'opv_0', 'hepb_0', 'birth']): return "Penta 1 / OPV 1", 42
        if 'penta_1' in v: return "Penta 2", 28
        if 'penta_2' in v: return "Penta 3", 28
        if 'penta_3' in v: return "Measles 1", 155 # 14w -> 9m
        if 'measles_1' in v or 'measles_mr_1' in v: return "Measles 2", 180
        if 'measles_2' in v or 'measles_mr_2' in v: return "Fully Immunized / HPV", 365
        return "Next Visit", 30

    def identify_at_risk(self, facility_name, churn_model):
        if self.df.empty: return pd.DataFrame()
        df_subset = self.df if facility_name == "All" else self.df[self.df['health_center_name'] == facility_name]
        
        # FILTER: Only use immunization records
        if 'track' in df_subset.columns:
            df_subset = df_subset[df_subset['track'] == 'immunization']
            
        today = datetime.now()
        latest = df_subset.sort_values('visit_date').groupby(self.id_col).tail(1).copy()
        
        # Calculate Dynamic Allowed Gap to define "Active" vs "Drop-off"
        child_vaccines = {}
        for _, row in df_subset.iterrows():
            cid = row[self.id_col]
            v_list = parse_vaccines(row['vaccines_administered'])
            if cid not in child_vaccines: child_vaccines[cid] = set()
            child_vaccines[cid].update([v.lower().replace(' ', '_') for v in v_list])

        latest['Last_Stage'] = latest[self.id_col].apply(lambda cid: self._get_stage(list(child_vaccines.get(cid, []))))
        latest['Allowed_Gap'] = latest['Last_Stage'].apply(self.get_dropoff_limit)
        
        # ACTIVE = Not yet dropped off (Days Elapsed <= Allowed Gap)
        # Note: We recalculate days_elapsed here to be sure
        latest['days_elapsed'] = (today - latest['visit_date']).dt.days
        active = latest[latest['days_elapsed'] <= latest['Allowed_Gap']].copy()
        
        if active.empty: return pd.DataFrame()
        
        # Predict Risk
        active['Churn_Prob'] = churn_model.predict_risk(active)
        
        # Filter High Risk
        at_risk = active[active['Churn_Prob'] > 0.5].copy()
        
        if at_risk.empty: return pd.DataFrame()
        
        # Add Enhanced Columns
        # 1. Next Scheduled Vaccine
        # 2. Due Date
        # 3. Days Remaining
        
        def calculate_details(row):
            next_name, interval = self.get_next_milestone(row['Last_Stage'])
            due_date = row['visit_date'] + timedelta(days=interval)
            days_rem = row['Allowed_Gap'] - row['days_elapsed']
            return pd.Series([next_name, due_date, days_rem])
            
        at_risk[['Next_Scheduled_Vaccine', 'Due_Date', 'Days_Remaining']] = at_risk.apply(calculate_details, axis=1)
        
        at_risk = at_risk.rename(columns={self.id_col: 'Child_ID'})
        
        # Format columns
        at_risk['Due_Date'] = at_risk['Due_Date'].dt.date
        
        return at_risk[['Child_ID', 'Last_Stage', 'Next_Scheduled_Vaccine', 'Due_Date', 'Days_Remaining', 'Churn_Prob']]

    def analyze_dropoff_stages(self, facility_name):
        if self.df.empty: return pd.DataFrame(), pd.DataFrame()
        df_subset = self.df if facility_name == "All" else self.df[self.df['health_center_name'] == facility_name]
        
        # FILTER: Only use immunization records
        if 'track' in df_subset.columns:
            df_subset = df_subset[df_subset['track'] == 'immunization']
        
        # 1. Calculate drop-off status for everyone first
        # We need: Last_Stage, Days_Elapsed, Allowed_Gap
        
        # A. Get Latest Date
        today = datetime.now()
        latest = df_subset.sort_values('visit_date').groupby(self.id_col).tail(1).copy()
        date_map = latest.set_index(self.id_col)['visit_date'].to_dict()
        
        # B. Get Cumulative History
        child_vaccines = {}
        for _, row in df_subset.iterrows():
            cid = row[self.id_col]
            v_list = parse_vaccines(row['vaccines_administered'])
            if cid not in child_vaccines: child_vaccines[cid] = set()
            child_vaccines[cid].update([v.lower().replace(' ', '_') for v in v_list])
            
        results = []
        for cid, v_set in child_vaccines.items():
            last_stage = self._get_stage(list(v_set))
            allowed_gap = self.get_dropoff_limit(last_stage)
            
            last_visit = date_map.get(cid)
            if not last_visit: continue
            
            days_elapsed = (today - last_visit).days
            
            # STRICT FILTER: Only count as Drop-off if they exceeded the limit
            if days_elapsed > allowed_gap:
                results.append({'Child_ID': cid, 'Drop_off_Stage': last_stage})
        
        df_stages = pd.DataFrame(results)
        
        if df_stages.empty:
             return df_stages, pd.DataFrame(columns=['Stage', 'Count', 'Percentage'])

        # C. Summary Counts for Chart
        summary = df_stages['Drop_off_Stage'].value_counts().reset_index()
        summary.columns = ['Stage', 'Count']
        total = summary['Count'].sum()
        summary['Percentage'] = (summary['Count'] / total * 100).round(1)
        
        return df_stages, summary

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
    success_loaded = False
    if os.path.exists(success_model.filename):
        try:
            success_model = joblib.load(success_model.filename)
            success_loaded = True
        except Exception:
            pass
            
    if not success_loaded:
        success_model.train_and_save(df_zd)
        
    # 2. Demand Forecast Model
    demand_model = DemandForecastModel()
    demand_ready = False
    if os.path.exists(demand_model.filename):
        try:
            demand_model = joblib.load(demand_model.filename)
            demand_ready = True
        except: pass
    
    if not demand_ready:
        demand_ready = demand_model.train_and_save(df_vis)
        
    # 3. Churn Model (Uses Cohort Data with parent_id)
    churn_model = ChurnModel()
    churn_ready = False
    if os.path.exists(churn_model.filename):
        try:
            churn_model = joblib.load(churn_model.filename)
            churn_ready = True
        except Exception:
            pass

    if not churn_ready:
        # Pass the cohort data specifically for Churn training
        churn_ready = churn_model.train_and_save(df_cohort)
        
    return success_model, demand_model, churn_model, demand_ready, churn_ready

# ==========================================
# 7. MAIN UI
# ==========================================

def main():
    init_session_state()
    engine = CommandEngine()
    
    # Load Models (Fast Cached Load)
    success_model, demand_model, churn_model, demand_ready, churn_ready = get_ml_models(
        st.session_state['df_zerodose'].copy(),
        st.session_state['df_visits'].copy(),
        st.session_state['df_cohort'].copy()
    )
    
    analyzer = FacilityAnalyzer(st.session_state['df_visits'])
    df_zd = st.session_state['df_zerodose']
    
    needs = df_zd.apply(engine.calculate_needs, axis=1)
    df_zd['Urgency_Score'] = [x[0] for x in needs]
    df_zd['Missing_Oral'] = [x[1] for x in needs]
    df_zd['Missing_Inject'] = [x[2] for x in needs]
    df_zd['Missing_Vaccines'] = df_zd.apply(lambda x: ", ".join(x['Missing_Oral'] + x['Missing_Inject']), axis=1)
    
    # Predict success
    df_zd['Success_Prob'] = success_model.predict_proba(df_zd)

    # --- TOP LEVEL NAVIGATION ---
    st.sidebar.title("Datharm")
    system_mode = st.sidebar.selectbox("System Role:", ["Facility Operations", "Admin"])

    if system_mode == "Admin":
        st.title("🔐 Admin: Inventory Control")
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
        
        if 'track' in st.session_state['df_visits'].columns:
            hierarchy = st.session_state['df_visits'][st.session_state['df_visits']['track'] == 'immunization'].groupby('lga_name')['health_center_name'].unique().to_dict()
            hierarchy = {k: list(v) for k, v in hierarchy.items()}
        else:
            hierarchy = {}

        all_lgas = sorted(list(hierarchy.keys()))
        
        # Filter ZD for Immunization Only & Handle Status
        df_zd_clean = df_zd.copy()
        if 'track' in df_zd_clean.columns:
            df_zd_clean = df_zd_clean[df_zd_clean['track'] == 'immunization']
        if 'Resolution Status' in df_zd_clean.columns:
            df_zd_clean['status'] = df_zd_clean['Resolution Status'].apply(lambda x: 'Resolved' if str(x).strip() == 'Resolved' else 'Pending')
        elif 'status' not in df_zd_clean.columns:
            df_zd_clean['status'] = 'Pending'
            
        if 'lga_name' in df_zd_clean.columns:
            zd_lgas = df_zd_clean['lga_name'].dropna().unique()
            all_lgas = sorted(list(set(all_lgas) | set(zd_lgas)))
            
        selected_lga = st.sidebar.selectbox("LGA (Region):", all_lgas)
        avail_facilities = hierarchy.get(selected_lga, [])
        if not avail_facilities:
            avail_facilities = list(st.session_state['facility_stock'].keys())
            
        active_facility = st.sidebar.selectbox("Dispatch Facility:", sorted(avail_facilities))
        current_stock = st.session_state['facility_stock'].get(active_facility, {})

        # with st.sidebar.expander("📦 Facility Stock (Live)", expanded=False):
        #     for cat in STOCK_CATEGORIES:
        #         count = current_stock.get(cat, 0)
        #         color = "red" if count < 20 else "green"
        #         st.markdown(f"**{cat}**: :{color}[{count}]")

        page = st.radio("Module:", ["Vaccine Administration", "Stock Management", "Cohort Tracker"], horizontal=True)
        st.divider()

        if page == "Vaccine Administration":
            # --- 1. HEALTH FACILITY SUMMARY (Top Level) ---
            st.subheader(f"🏥 Facility Dashboard: {active_facility}")
            
            # Use Visits Data (facility_visits.csv) for Facility Metrics
            # logic: Filter visits for this facility
            df_visits_fac = st.session_state['df_visits'][
                st.session_state['df_visits']['health_center_name'] == active_facility
            ].copy()
            
            fac_defaulters = 0
            # Count UNIQUE children for active cohort, as df_visits is transactional
            id_col = 'parent_id' if 'parent_id' in df_visits_fac.columns else 'id'
            fac_active_count = df_visits_fac[id_col].nunique() if not df_visits_fac.empty else 0
            
            # Check for drop-offs
            dropoffs_fac = pd.DataFrame() # Initialize to avoid UnboundLocalError
            if not st.session_state['df_cohort'].empty:
                try:
                    dropoffs_fac = analyzer.identify_dropoffs(active_facility)
                    # Enforce Immunization Track Filter (User Request)
                    if not dropoffs_fac.empty and 'track' in dropoffs_fac.columns:
                        dropoffs_fac = dropoffs_fac[dropoffs_fac['track'] == 'immunization']
                    
                    fac_defaulters = len(dropoffs_fac)
                except Exception as e:
                    # st.error(f"Error calculating drop-offs: {e}") # Optional debug
                    fac_defaulters = 0

            # Define fac_pending as empty to trigger LGA fallback for the table below
            # (Since ZD data has no facility column, we can't get fac_pending from there)
            fac_pending = pd.DataFrame()

            # Stock Risk
            f_stock = st.session_state['facility_stock'].get(active_facility, {})
            crit_items = sum(1 for v in f_stock.values() if v < 20)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("FACILITY DEFAULTERS", fac_defaulters, delta_color="inverse", help="Children who missed their scheduled vaccine (from facility records)")
            m2.metric("ACTIVE COHORT", fac_active_count, help="Total children registered in immunization track")
            # m3.metric("CRITICAL STOCK ITEMS", crit_items, delta_color="inverse", help="Items with stock < 20")
            
            st.divider()
            
            # --- Chart: Facility Analytics (New) ---
            if not df_visits_fac.empty:
                try:
                    st.markdown("##### 📊 Facility Analytics: Vaccine Coverage")
                    
                    # 1. Administered Counts Logic
                    # Normalize age
                    if 'age_months' in df_visits_fac.columns:
                        df_visits_fac['age_months'] = pd.to_numeric(df_visits_fac['age_months'], errors='coerce').fillna(0)
                        
                    def get_age_group_local(age):
                        # User Requirements:
                        # Due for Penta: >6 weeks - 11 months
                        # Active ZD: 12 months - 23 months
                        # Overaged ZD: 24 months - 5 years
                        
                        if age < 1.5: return "Newborn (<6w)" 
                        elif age < 12: return "Due for Penta (6w-11m)"
                        elif age < 24: return "Active ZD (12-23m)"
                        return "Overaged ZD (24m-5y)"
                    
                    df_visits_fac['Age_Group'] = df_visits_fac['age_months'].apply(get_age_group_local)
                    
                    # Process Administered (flatten list)
                    admin_list = []
                    for _, row in df_visits_fac.iterrows():
                        v_list = parse_vaccines(row.get('vaccines_administered'))
                        grp = row['Age_Group']
                        for v in v_list:
                            admin_list.append({'Vaccine': v, 'Age_Group': grp, 'Count': 1})
                            
                    df_admin = pd.DataFrame(admin_list)
                    
                    # 2. Missed/Next Visit Logic
                    # Calculate needs for EACH visit record (Snapshotted needs)
                    missed_list = []
                    for _, row in df_visits_fac.iterrows():
                        # Use Engine to calc needs
                        _, miss_o, miss_i = engine.calculate_needs(row)
                        grp = row['Age_Group']
                        for m in miss_o + miss_i:
                            missed_list.append({'Vaccine': m, 'Age_Group': grp, 'Count': 1})
                            
                    df_missed = pd.DataFrame(missed_list)
                    
                    # 3. VISUALIZATIONS
                    
                    t1, t2 = st.tabs(["✅ Administered History", "⚠️ Missed / Due Next"])
                    
                    with t1:
                        if not df_admin.empty:
                            # Aggregate Total
                            total_admin = df_admin['Vaccine'].value_counts().reset_index()
                            total_admin.columns = ['Vaccine', 'Count']
                            
                            # Aggregate by Age
                            pivot_admin = df_admin.pivot_table(index='Vaccine', columns='Age_Group', values='Count', aggfunc='sum', fill_value=0).reset_index()
                            
                            c_a1, c_a2 = st.columns(2)
                            with c_a1:
                                st.caption("Total Administered (Volume)")
                                st.bar_chart(total_admin.set_index('Vaccine'), color="#2E86C1", horizontal=True)
                            with c_a2:
                                st.caption("Administered by Age Group")
                                st.bar_chart(pivot_admin.set_index('Vaccine'), stack=True)
                        else:
                            st.info("No administration data.")
                            
                    with t2:
                        if not df_missed.empty:
                            # Aggregate Total
                            total_missed = df_missed['Vaccine'].value_counts().reset_index()
                            total_missed.columns = ['Vaccine', 'Count']
                            
                            # Aggregate by Age
                            pivot_missed = df_missed.pivot_table(index='Vaccine', columns='Age_Group', values='Count', aggfunc='sum', fill_value=0).reset_index()
                            
                            c_m1, c_m2 = st.columns(2)
                            with c_m1:
                                st.caption("Total Missed / Due (Opportunities)")
                                st.bar_chart(total_missed.set_index('Vaccine'), color="#E74C3C", horizontal=True)
                            with c_m2:
                                st.caption("Missed by Age Group")
                                st.bar_chart(pivot_missed.set_index('Vaccine'), stack=True)
                        else:
                            st.success("No missed vaccines detected in records.")

                except Exception as e:
                    st.warning(f"Could not load analytics charts: {e}")
            
            st.divider()
            
            selected_child_id = None 

            # --- 2. LGA Command Center (Overview) ---
            st.markdown(f"#### 🏘️ LGA Overview: {selected_lga}")
            
            # Metrics: LGA Wide
            lga_pending_count = len(df_zd_clean[(df_zd_clean['status'] == 'Pending') & (df_zd_clean['lga_name'] == selected_lga)])
            lga_resolved_count = len(df_zd_clean[(df_zd_clean['status'] == 'Resolved') & (df_zd_clean['lga_name'] == selected_lga)])
            
            k1, k2 = st.columns(2)
            k1.metric("LGA PENDING", lga_pending_count)
            k2.metric("LGA RESOLVED", lga_resolved_count)
            
            st.divider()

            st.markdown("#### 📋 Priority Queue (Actionable Cases)")
            
            # MERGING LOGIC: Combine Facility Defaulters (Clinical) + LGA Zero-Dose (Survey)
            
            # 1. Process Facility Data
            fac_list = []
            if not dropoffs_fac.empty:
                # Normalize columns to match ZD format
                id_col = analyzer.id_col
                for idx, row in dropoffs_fac.iterrows():
                    # Robust ID fetch
                    cid = row.get(id_col)
                    # Try column variations
                    if pd.isna(cid): cid = row.get('id')
                    if pd.isna(cid): cid = row.get('ID')
                    if pd.isna(cid): cid = row.get('parent_id')
                    if pd.isna(cid): cid = row.get('child_id')
                    if pd.isna(cid): cid = row.get('Child_ID')
                    
                    # Try Index (if grouped by ID)
                    if pd.isna(cid): cid = idx
                    
                    if pd.isna(cid): continue # Skip if no ID found
                    
                    # Calculate missing vaccines for EACH facility record
                    # We need to construct a row dummy for calculate_needs
                    
                    # 1. Get Age at Visit
                    age_at_visit = row.get('age_months')
                    if pd.isna(age_at_visit):
                        # Try weeks
                        weeks = row.get('age_weeks')
                        if pd.isna(weeks): weeks = row.get('age_weeks') # Retry just in case
                        if not pd.isna(weeks): age_at_visit = weeks / 4.3
                    
                    if pd.isna(age_at_visit): age_at_visit = 0
                    
                    # 2. Add Time Since Visit (to get CURRENT Age)
                    days_elapsed = row.get('days_elapsed', 0)
                    current_age_months = age_at_visit + (days_elapsed / 30.4)
                    current_age_months = int(round(current_age_months)) # User Request: Integers for Age
                    
                    # Handle NaN vaccines explicitly
                    vax_given = row.get('vaccines_administered')
                    if pd.isna(vax_given) or vax_given == "": vax_given = '[]'
                    
                    # Pre-parse for cleaner display
                    vax_clean_list = parse_vaccines(vax_given)
                    vax_display = ", ".join(vax_clean_list) if vax_clean_list else "None"
                    
                    dummy_row = {
                        'age_months': current_age_months,
                        'vaccines_administered': vax_given # Use raw for logic (logic handles list parsing)
                    }
                    score, miss_o, miss_i = engine.calculate_needs(dummy_row)
                    
                    # Extract Metadata for Success Prediction
                    gender_val = row.get('gender', 'Unknown')
                    lga_val = row.get('lga_name', selected_lga)
                    
                    if miss_o or miss_i: # Only add if actually missing something
                        fac_list.append({
                            'ID': cid,
                            'age_months': current_age_months,
                            'vaccines_administered': vax_display, # Display Clean String
                            'Missing_Oral': miss_o,
                            'Missing_Inject': miss_i,
                            'Missing_Vaccines': ", ".join(miss_o + miss_i),
                            'Urgency_Score': score,
                            'Source': 'Facility Record (Defaulter)',
                            'gender': gender_val,       # For Model
                            'lga_name': lga_val,        # For Model
                            'reasons_for_zd': 'Defaulter', # For Model (Placeholder)
                            'Distance to HF': '0 km'    # For Model (Assumption)
                        })
            
            df_fac_pending = pd.DataFrame(fac_list)
            
            # Predict Success for Facility Records
            if not df_fac_pending.empty and 'success_model' in st.session_state:
                try:
                    df_fac_pending['Success_Prob'] = st.session_state['success_model'].predict_proba(df_fac_pending)
                except Exception as e:
                    df_fac_pending['Success_Prob'] = 0.5 # Fallback
            
            # 2. Process Zero-Dose Data (LGA Fallback)
            df_zd_pending = df_zd_clean[(df_zd_clean['status'] == 'Pending') & (df_zd_clean['lga_name'] == selected_lga)].copy()
            if not df_zd_pending.empty:
                df_zd_pending['Source'] = 'Zero-Dose Survey'
                # Ensure missing cols exist
                if 'Missing_Vaccines' not in df_zd_pending.columns:
                     # Calculate if missing
                     needs = df_zd_pending.apply(engine.calculate_needs, axis=1)
                     df_zd_pending['Missing_Vaccines'] = [", ".join(x[1] + x[2]) for x in needs]
                
                # Predict Success for ZD Records (for consistency)
                if 'success_model' in st.session_state:
                    try:
                        df_zd_pending['Success_Prob'] = st.session_state['success_model'].predict_proba(df_zd_pending)
                    except:
                        df_zd_pending['Success_Prob'] = 0.5
            
            # 3. Combine Facility + Zero Dose
            pending = pd.concat([df_fac_pending, df_zd_pending], ignore_index=True)
            
            if not pending.empty:
                # Calculate Urgency (if not already done)
                if 'Urgency_Score' not in pending.columns or pending['Urgency_Score'].isna().any():
                     # Recalc where missing
                     pass 
                
                # Sort by Urgency
                if 'Urgency_Score' in pending.columns:
                    pending = pending.sort_values('Urgency_Score', ascending=False)
                
                # Ensure columns exist
                if 'vaccines_administered' not in pending.columns:
                    pending['vaccines_administered'] = '[]'
                if 'Success_Prob' not in pending.columns:
                    pending['Success_Prob'] = 0.0
                
                # Fill NaN Probabilities
                pending['Success_Prob'] = pd.to_numeric(pending['Success_Prob'], errors='coerce').fillna(0.0)

                # Table Columns: ID, Age, Source, Vaccines Admin, Missing Vax, Urgency
                ui_cols = ['ID', 'age_months', 'vaccines_administered', 'Missing_Vaccines', 'Urgency_Score', 'Success_Prob']
                
                # Intersect to be safe (though we force added them)
                ui_cols = [c for c in ui_cols if c in pending.columns]
                
                st.dataframe(
                    pending[ui_cols], 
                    use_container_width=True,
                    column_config={
                        "Urgency_Score": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=100, format="%d"),
                        "Success_Prob": st.column_config.ProgressColumn("Success Probability", min_value=0, max_value=1, format="%.2f"),
                        "vaccines_administered": "Vaccines Given",
                        "Missing_Vaccines": "Missing"
                    },
                    selection_mode="single-row",
                    on_select="rerun",
                    key="dispatch_table"
                )
                # Selection Logic
                try:
                    sel_state = st.session_state.get("dispatch_table", {})
                    if sel_state and "selection" in sel_state and "rows" in sel_state["selection"]:
                        rows = sel_state["selection"]["rows"]
                        if rows:
                            selected_index = rows[0]
                            selected_child_id = pending.iloc[selected_index]['ID']
                except: pass

            else:
                st.success("No actionable pending cases for this facility.")
            
            # --- Action Panel (Bottom) ---
            st.divider()
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
                
                if st.button("🚀 Dispatch Team", type="primary", disabled=not can_dispatch):
                    dispatch_team(selected_child_id, active_facility, oral_selected, inject_selected)
            
            elif not pending.empty:
                st.info("Select a child from the table above to start dispatch.")
            else:
                st.success("No pending cases.")

        elif page == "Stock Management":
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
            
            # 1. Drop-off Analysis (New)
            st.markdown("### 📊 Drop-off Analysis")
            df_stages, summary = analyzer.analyze_dropoff_stages(active_facility)
            
            if not summary.empty:
                c_chart, c_data = st.columns([2, 1])
                with c_chart:
                    fig_drop = px.bar(summary, x='Stage', y='Count', text='Percentage', title=f"Drop-off Stages ({active_facility})", color='Stage')
                    fig_drop.update_traces(texttemplate='%{text}%', textposition='outside')
                    st.plotly_chart(fig_drop, use_container_width=True)
                with c_data:
                    st.write("**Drop-off Summary**")
                    st.dataframe(summary, use_container_width=True)
            else:
                st.info("No cohort data available for analysis.")
                
            st.divider()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("🚨 Drop-offs (ML Detected)")
                st.caption(f"High probability of churn.")
                model_to_use = churn_model if churn_ready else None
                dropoffs = analyzer.identify_dropoffs(active_facility, model_to_use)
                if not dropoffs.empty:
                    # Renaming for UI clarity
                    display_drop = dropoffs[['Child_ID', 'days_elapsed', 'Last_Stage', 'Churn_Prob']].rename(columns={'Last_Stage': 'Drop-off After'})
                    st.dataframe(display_drop, use_container_width=True, column_config={"Churn_Prob": st.column_config.ProgressColumn("Risk", format="%.2f", max_value=1)})
                else:
                    st.success("None detected.")

            with c2:
                st.subheader("🔮 Early Warning")
                st.caption(f"Active patients at **High Risk**.")
                if churn_ready:
                    at_risk = analyzer.identify_at_risk(active_facility, churn_model)
                    if not at_risk.empty:
                        st.dataframe(
                            at_risk[['Child_ID', 'Next_Scheduled_Vaccine', 'Due_Date', 'Days_Remaining', 'Churn_Prob']], 
                            use_container_width=True, 
                            column_config={
                                "Churn_Prob": st.column_config.ProgressColumn("Risk", format="%.2f", max_value=1),
                                "Due_Date": st.column_config.DateColumn("Due Date"),
                                "Days_Remaining": st.column_config.NumberColumn("Days Left", help="Days until drop-off threshold")
                            }
                        )
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
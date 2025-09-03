import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import random
from datetime import datetime, timedelta

class HealthcareDataSimulator:
    """Simulates healthcare data for federated learning experiments"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.medical_conditions = [
            'diabetes', 'hypertension', 'heart_disease', 'respiratory_disease',
            'cancer', 'kidney_disease', 'neurological_disorder', 'mental_health'
        ]
        
    def generate_patient_data(self, num_patients: int, facility_specialty: str = 'general') -> pd.DataFrame:
        """Generate synthetic patient data for a healthcare facility"""
        
        patients = []
        for i in range(num_patients):
            patient = self._generate_single_patient(i, facility_specialty)
            patients.append(patient)
        
        return pd.DataFrame(patients)
    
    def _generate_single_patient(self, patient_id: int, facility_specialty: str) -> Dict[str, Any]:
        """Generate data for a single patient"""
        
        # Basic demographics
        age = max(18, int(np.random.normal(50, 20)))
        gender = random.choice(['M', 'F'])
        
        # Vital signs (with some correlation to age and conditions)
        base_systolic = 120 + (age - 50) * 0.5 + np.random.normal(0, 15)
        systolic_bp = max(80, min(200, base_systolic))
        diastolic_bp = max(50, min(120, systolic_bp * 0.65 + np.random.normal(0, 8)))
        
        heart_rate = max(50, min(120, 72 + np.random.normal(0, 12)))
        temperature = 36.5 + np.random.normal(0, 0.8)
        
        # Lab values
        glucose = max(70, 100 + np.random.normal(0, 30))
        cholesterol = max(120, 200 + np.random.normal(0, 40))
        hemoglobin = max(8, 14 + np.random.normal(0, 2))
        
        # Lifestyle factors
        bmi = max(15, 25 + np.random.normal(0, 5))
        smoking = random.choice([0, 1])  # 0 = No, 1 = Yes
        exercise_hours = max(0, np.random.exponential(3))
        
        # Generate condition based on facility specialty
        primary_condition = self._assign_condition(facility_specialty, age, bmi, systolic_bp, glucose)
        
        # Risk scores (derived from other features)
        cardiovascular_risk = self._calculate_cv_risk(age, gender, systolic_bp, cholesterol, smoking)
        diabetes_risk = self._calculate_diabetes_risk(age, bmi, glucose)
        
        return {
            'patient_id': f'P_{patient_id:05d}',
            'age': age,
            'gender': gender,
            'systolic_bp': round(systolic_bp, 1),
            'diastolic_bp': round(diastolic_bp, 1),
            'heart_rate': round(heart_rate, 1),
            'temperature': round(temperature, 1),
            'glucose': round(glucose, 1),
            'cholesterol': round(cholesterol, 1),
            'hemoglobin': round(hemoglobin, 1),
            'bmi': round(bmi, 1),
            'smoking': smoking,
            'exercise_hours': round(exercise_hours, 1),
            'primary_condition': primary_condition,
            'cardiovascular_risk': round(cardiovascular_risk, 3),
            'diabetes_risk': round(diabetes_risk, 3),
            'admission_date': self._generate_admission_date(),
            'facility_specialty': facility_specialty
        }
    
    def _assign_condition(self, specialty: str, age: int, bmi: float, bp: float, glucose: float) -> str:
        """Assign primary condition based on facility specialty and patient characteristics"""
        
        # Base probabilities for different conditions
        condition_probs = {
            'diabetes': 0.15,
            'hypertension': 0.25,
            'heart_disease': 0.15,
            'respiratory_disease': 0.10,
            'cancer': 0.08,
            'kidney_disease': 0.07,
            'neurological_disorder': 0.10,
            'mental_health': 0.10
        }
        
        # Adjust probabilities based on specialty
        specialty_adjustments = {
            'cardiology': {'heart_disease': 2.5, 'hypertension': 2.0, 'diabetes': 1.5},
            'endocrinology': {'diabetes': 3.0, 'hypertension': 1.5},
            'oncology': {'cancer': 4.0},
            'neurology': {'neurological_disorder': 3.0},
            'nephrology': {'kidney_disease': 3.0, 'diabetes': 1.8, 'hypertension': 1.8},
            'pulmonology': {'respiratory_disease': 3.0},
            'psychiatry': {'mental_health': 4.0},
            'general': {}  # No adjustments for general practice
        }
        
        # Apply specialty adjustments
        if specialty in specialty_adjustments:
            for condition, multiplier in specialty_adjustments[specialty].items():
                condition_probs[condition] *= multiplier
        
        # Adjust based on patient characteristics
        if age > 65:
            condition_probs['heart_disease'] *= 1.5
            condition_probs['hypertension'] *= 1.8
            condition_probs['diabetes'] *= 1.3
        
        if bmi > 30:
            condition_probs['diabetes'] *= 1.8
            condition_probs['heart_disease'] *= 1.4
        
        if bp > 140:
            condition_probs['hypertension'] *= 2.0
        
        if glucose > 125:
            condition_probs['diabetes'] *= 2.0
        
        # Normalize probabilities
        total_prob = sum(condition_probs.values())
        normalized_probs = {k: v/total_prob for k, v in condition_probs.items()}
        
        # Select condition based on probabilities
        conditions = list(normalized_probs.keys())
        probabilities = list(normalized_probs.values())
        
        return np.random.choice(conditions, p=probabilities)
    
    def _calculate_cv_risk(self, age: int, gender: str, systolic_bp: float, 
                          cholesterol: float, smoking: int) -> float:
        """Calculate cardiovascular risk score"""
        risk_score = 0
        
        # Age component
        if gender == 'M':
            risk_score += max(0, (age - 45) * 0.02)
        else:
            risk_score += max(0, (age - 55) * 0.02)
        
        # Blood pressure component
        if systolic_bp > 140:
            risk_score += 0.1
        elif systolic_bp > 120:
            risk_score += 0.05
        
        # Cholesterol component
        if cholesterol > 240:
            risk_score += 0.08
        elif cholesterol > 200:
            risk_score += 0.04
        
        # Smoking component
        if smoking:
            risk_score += 0.15
        
        return min(1.0, risk_score)
    
    def _calculate_diabetes_risk(self, age: int, bmi: float, glucose: float) -> float:
        """Calculate diabetes risk score"""
        risk_score = 0
        
        # Age component
        if age > 45:
            risk_score += 0.1
        if age > 65:
            risk_score += 0.1
        
        # BMI component
        if bmi > 30:
            risk_score += 0.2
        elif bmi > 25:
            risk_score += 0.1
        
        # Glucose component
        if glucose > 125:
            risk_score += 0.3
        elif glucose > 100:
            risk_score += 0.15
        
        return min(1.0, risk_score)
    
    def _generate_admission_date(self) -> str:
        """Generate a random admission date within the last year"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        random_date = start_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        return random_date.strftime('%Y-%m-%d %H:%M')
    
    def generate_federated_datasets(self, facilities_config: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Generate datasets for multiple healthcare facilities"""
        
        datasets = {}
        
        for i, config in enumerate(facilities_config):
            facility_id = config.get('id', f'facility_{i}')
            num_patients = config.get('num_patients', random.randint(100, 500))
            specialty = config.get('specialty', 'general')
            
            # Add some variation in data distribution
            data_quality_factor = config.get('data_quality', random.uniform(0.8, 1.0))
            
            dataset = self.generate_patient_data(num_patients, specialty)
            
            # Simulate data quality variations
            if data_quality_factor < 1.0:
                dataset = self._add_data_quality_issues(dataset, data_quality_factor)
            
            datasets[facility_id] = dataset
        
        return datasets
    
    def _add_data_quality_issues(self, dataset: pd.DataFrame, quality_factor: float) -> pd.DataFrame:
        """Add realistic data quality issues to simulate real-world scenarios"""
        
        dataset_copy = dataset.copy()
        
        # Add missing values
        missing_rate = 1 - quality_factor
        for column in ['cholesterol', 'hemoglobin', 'exercise_hours']:
            missing_mask = np.random.random(len(dataset_copy)) < missing_rate * 0.1
            dataset_copy.loc[missing_mask, column] = np.nan
        
        # Add measurement noise
        noise_columns = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'glucose']
        for column in noise_columns:
            if column in dataset_copy.columns:
                noise = np.random.normal(0, dataset_copy[column].std() * (1 - quality_factor) * 0.1)
                dataset_copy[column] += noise
        
        # Add occasional outliers
        outlier_rate = (1 - quality_factor) * 0.05
        for column in noise_columns:
            if column in dataset_copy.columns:
                outlier_mask = np.random.random(len(dataset_copy)) < outlier_rate
                outliers = dataset_copy[column].mean() + np.random.normal(0, dataset_copy[column].std() * 3, outlier_mask.sum())
                dataset_copy.loc[outlier_mask, column] = outliers
        
        return dataset_copy
    
    def create_ml_features(self, dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert healthcare data to ML-ready features and labels"""
        
        # Select numerical features
        feature_columns = [
            'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature',
            'glucose', 'cholesterol', 'hemoglobin', 'bmi', 'smoking', 'exercise_hours'
        ]
        
        # Handle missing values
        features_df = dataset[feature_columns].copy()
        features_df = features_df.fillna(features_df.mean())
        
        # Convert to numpy array
        X = features_df.values
        
        # Create binary labels for high-risk patients (simplified)
        # High risk defined as cardiovascular_risk > 0.3 OR diabetes_risk > 0.3
        y = ((dataset['cardiovascular_risk'] > 0.3) | (dataset['diabetes_risk'] > 0.3)).astype(int).values
        
        return X, y
    
    def get_data_statistics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of the dataset"""
        
        stats = {
            'num_patients': len(dataset),
            'age_distribution': {
                'mean': dataset['age'].mean(),
                'std': dataset['age'].std(),
                'min': dataset['age'].min(),
                'max': dataset['age'].max()
            },
            'gender_distribution': dataset['gender'].value_counts().to_dict(),
            'condition_distribution': dataset['primary_condition'].value_counts().to_dict(),
            'risk_metrics': {
                'mean_cv_risk': dataset['cardiovascular_risk'].mean(),
                'mean_diabetes_risk': dataset['diabetes_risk'].mean(),
                'high_risk_patients': len(dataset[
                    (dataset['cardiovascular_risk'] > 0.3) | (dataset['diabetes_risk'] > 0.3)
                ])
            },
            'data_quality': {
                'missing_values': dataset.isnull().sum().to_dict(),
                'completeness': (1 - dataset.isnull().sum() / len(dataset)).to_dict()
            }
        }
        
        return stats
    
    def simulate_data_heterogeneity(self, base_dataset: pd.DataFrame, heterogeneity_level: float = 0.3) -> pd.DataFrame:
        """Simulate data heterogeneity across different facilities"""
        
        dataset_copy = base_dataset.copy()
        
        # Introduce feature drift
        if heterogeneity_level > 0:
            # Age distribution shift
            age_shift = np.random.normal(0, 5 * heterogeneity_level, len(dataset_copy))
            dataset_copy['age'] = np.clip(dataset_copy['age'] + age_shift, 18, 100)
            
            # Systematic measurement bias
            bp_bias = np.random.normal(0, 10 * heterogeneity_level)
            dataset_copy['systolic_bp'] += bp_bias
            dataset_copy['diastolic_bp'] += bp_bias * 0.6
            
            # Population-specific risk factor prevalence
            smoking_adjustment = heterogeneity_level * (np.random.random() - 0.5)
            smoking_mask = np.random.random(len(dataset_copy)) < abs(smoking_adjustment)
            if smoking_adjustment > 0:
                dataset_copy.loc[smoking_mask, 'smoking'] = 1
            else:
                dataset_copy.loc[smoking_mask, 'smoking'] = 0
        
        return dataset_copy


class PrivacyPreservingDataProcessor:
    """Handles privacy-preserving preprocessing of healthcare data"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        
    def add_differential_privacy(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add differential privacy noise to numerical data"""
        noise_scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, noise_scale, data.shape)
        return data + noise
    
    def k_anonymize_categorical(self, data: pd.Series, k: int = 5) -> pd.Series:
        """Apply k-anonymity to categorical data"""
        value_counts = data.value_counts()
        
        # Group small categories into "Other"
        small_categories = value_counts[value_counts < k].index
        anonymized_data = data.copy()
        anonymized_data[anonymized_data.isin(small_categories)] = 'Other'
        
        return anonymized_data
    
    def generalize_numerical(self, data: pd.Series, num_bins: int = 5) -> pd.Series:
        """Generalize numerical data into bins for privacy"""
        return pd.cut(data, bins=num_bins, labels=[f'Range_{i+1}' for i in range(num_bins)])
    
    def apply_local_differential_privacy(self, dataset: pd.DataFrame, 
                                       epsilon: float = 0.1) -> pd.DataFrame:
        """Apply local differential privacy to the entire dataset"""
        
        processed_dataset = dataset.copy()
        
        # Apply noise to numerical columns
        numerical_columns = [
            'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature',
            'glucose', 'cholesterol', 'hemoglobin', 'bmi', 'exercise_hours'
        ]
        
        for column in numerical_columns:
            if column in processed_dataset.columns:
                sensitivity = processed_dataset[column].std() * 0.1  # Conservative sensitivity
                noisy_data = self.add_differential_privacy(
                    processed_dataset[column].values, sensitivity
                )
                processed_dataset[column] = noisy_data
        
        # Generalize categorical data
        if 'age' in processed_dataset.columns:
            processed_dataset['age_group'] = self.generalize_numerical(
                processed_dataset['age'], num_bins=6
            )
        
        # Apply k-anonymity to sensitive categorical fields
        categorical_columns = ['primary_condition', 'facility_specialty']
        for column in categorical_columns:
            if column in processed_dataset.columns:
                processed_dataset[column] = self.k_anonymize_categorical(
                    processed_dataset[column], k=3
                )
        
        return processed_dataset

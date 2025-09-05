#!/usr/bin/env python3
"""
BPM Prediction Project - Main Execution Script

This script runs the complete BPM prediction pipeline including:
1. Data preprocessing
2. Feature engineering
3. Model training
4. Predictions and submission generation
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the training and test data"""
    print("🔄 Loading and preprocessing data...")
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv', comment='#')
    
    # Standardize column names
    test_column_mapping = {
        'AudioLoudness': 'audio_loudness',
        'VocalContent': 'vocal_content', 
        'AcousticQuality': 'acoustic_quality',
        'InstrumentalScore': 'instrumental_score',
        'LivePerformanceLikelihood': 'live_performance_likelihood',
        'MoodScore': 'mood_score',
        'Energy': 'energy',
        'RhythmScore': 'rhythm_score',
        'TrackDurationMs': 'track_duration_ms'
    }
    
    test_df = test_df.rename(columns=test_column_mapping)
    
    print(f"   ✅ Training data: {train_df.shape}")
    print(f"   ✅ Test data: {test_df.shape}")
    
    return train_df, test_df

def create_features(train_df, test_df):
    """Create engineered features for both datasets"""
    print("🔧 Creating engineered features...")
    
    def engineer_features(df):
        df_fe = df.copy()
        
        # Basic interaction features
        df_fe['Energy_Rhythm_Ratio'] = df_fe['energy'] / (df_fe['rhythm_score'] + 1e-6)
        df_fe['Vocal_Instrumental_Ratio'] = df_fe['vocal_content'] / (df_fe['instrumental_score'] + 1e-6)
        df_fe['Energy_Mood_Product'] = df_fe['energy'] * df_fe['mood_score']
        df_fe['Acoustic_Energy_Product'] = df_fe['acoustic_quality'] * df_fe['energy']
        
        # Duration-based features
        df_fe['Duration_Minutes'] = df_fe['track_duration_ms'] / 60000
        df_fe['Energy_per_Minute'] = df_fe['energy'] / (df_fe['Duration_Minutes'] + 1e-6)
        df_fe['Rhythm_per_Minute'] = df_fe['rhythm_score'] / (df_fe['Duration_Minutes'] + 1e-6)
        
        # Composite scores
        df_fe['Danceability_Score'] = (0.4 * df_fe['energy'] + 
                                      0.3 * df_fe['rhythm_score'] + 
                                      0.3 * (-df_fe['audio_loudness'] / 20))
        
        df_fe['Musical_Complexity'] = (df_fe['acoustic_quality'] + 
                                      df_fe['instrumental_score'] + 
                                      df_fe['vocal_content']) / 3
        
        # Non-linear transformations
        df_fe['Energy_Squared'] = df_fe['energy'] ** 2
        df_fe['Rhythm_Squared'] = df_fe['rhythm_score'] ** 2
        df_fe['Energy_Sqrt'] = np.sqrt(df_fe['energy'])
        
        return df_fe
    
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    
    print(f"   ✅ Enhanced training data: {train_fe.shape}")
    print(f"   ✅ Enhanced test data: {test_fe.shape}")
    
    return train_fe, test_fe

def train_models(X_train, y_train, X_val, y_val):
    """Train multiple models and return the best one"""
    print("🤖 Training models...")
    
    models = {
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    best_model = None
    best_score = float('inf')
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'model': model
        }
        
        print(f"   {name:15} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_name = name
    
    print(f"   🏆 Best model: {best_name} (RMSE: {best_score:.3f})")
    return best_model, results

def make_predictions(model, test_features, test_ids):
    """Make predictions and create submission file"""
    print("🔮 Making predictions...")
    
    predictions = model.predict(test_features)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': predictions
    })
    
    # Save submission
    submission.to_csv('data/submission_final.csv', index=False)
    
    print(f"   ✅ Predictions saved to data/submission_final.csv")
    print(f"   📊 Prediction statistics:")
    print(f"      Mean BPM: {predictions.mean():.2f}")
    print(f"      Std BPM: {predictions.std():.2f}")
    print(f"      Min BPM: {predictions.min():.2f}")
    print(f"      Max BPM: {predictions.max():.2f}")
    
    return submission

def main():
    """Main execution pipeline"""
    print("🎵 BPM PREDICTION PIPELINE")
    print("=" * 50)
    
    # 1. Load and preprocess data
    train_df, test_df = load_and_preprocess_data()
    
    # 2. Create features
    train_fe, test_fe = create_features(train_df, test_df)
    
    # 3. Prepare training data
    print("📊 Preparing training data...")
    non_feature_cols = ['id', 'title', 'artist', 'album', 'beats_per_minute']
    feature_cols = [col for col in train_fe.columns if col not in non_feature_cols]
    
    X = train_fe[feature_cols]
    y = train_fe['beats_per_minute']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   ✅ Features: {len(feature_cols)}")
    print(f"   ✅ Training samples: {len(X_train)}")
    print(f"   ✅ Validation samples: {len(X_val)}")
    
    # 4. Train models
    best_model, results = train_models(X_train, y_train, X_val, y_val)
    
    # 5. Make predictions
    test_features = test_fe[feature_cols]
    submission = make_predictions(best_model, test_features, test_df['id'])
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Files created:")
    print(f"   - data/submission_final.csv")
    print(f"   - data/train_experimental.csv (from experimental_approaches.py)")
    print(f"   - data/test_experimental.csv (from experimental_approaches.py)")
    
    return submission

if __name__ == "__main__":
    submission = main()

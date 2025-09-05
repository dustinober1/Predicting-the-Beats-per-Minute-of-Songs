#!/usr/bin/env python3
"""
BPM Prediction Project - Complete Evaluation and Summary

This script runs all components of the BPM prediction project and generates
a comprehensive summary of results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def run_all_components():
    """Run all project components and collect results"""
    print("🚀 RUNNING COMPLETE BPM PREDICTION PROJECT")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'components': {}
    }
    
    # 1. Run experimental approaches
    print("1️⃣ Running experimental approaches...")
    os.system("python experimental_approaches.py > logs_experimental.txt 2>&1")
    
    if os.path.exists('data/train_experimental.csv'):
        exp_train = pd.read_csv('data/train_experimental.csv')
        exp_test = pd.read_csv('data/test_experimental.csv')
        results['components']['experimental'] = {
            'status': 'SUCCESS',
            'train_shape': exp_train.shape,
            'test_shape': exp_test.shape,
            'features_created': exp_train.shape[1] - 14  # Original had 14 columns
        }
        print(f"   ✅ Experimental features created: {results['components']['experimental']['features_created']}")
    else:
        results['components']['experimental'] = {'status': 'FAILED'}
        print("   ❌ Experimental approaches failed")
    
    # 2. Run main pipeline
    print("\n2️⃣ Running main pipeline...")
    os.system("python run_pipeline.py > logs_pipeline.txt 2>&1")
    
    if os.path.exists('data/submission_final.csv'):
        submission = pd.read_csv('data/submission_final.csv')
        results['components']['main_pipeline'] = {
            'status': 'SUCCESS',
            'predictions_count': len(submission),
            'prediction_stats': {
                'mean': submission['BeatsPerMinute'].mean(),
                'std': submission['BeatsPerMinute'].std(),
                'min': submission['BeatsPerMinute'].min(),
                'max': submission['BeatsPerMinute'].max()
            }
        }
        print(f"   ✅ Main pipeline completed with {len(submission)} predictions")
    else:
        results['components']['main_pipeline'] = {'status': 'FAILED'}
        print("   ❌ Main pipeline failed")
    
    # 3. Check notebook execution (if run previously)
    if os.path.exists('data/submission.csv'):
        notebook_submission = pd.read_csv('data/submission.csv')
        results['components']['notebook'] = {
            'status': 'SUCCESS',
            'predictions_count': len(notebook_submission)
        }
        print(f"   ✅ Notebook predictions found: {len(notebook_submission)} predictions")
    else:
        results['components']['notebook'] = {'status': 'NOT_RUN'}
        print("   ℹ️  Notebook submission file not found (run notebooks manually)")
    
    return results

def analyze_data_quality():
    """Analyze the quality and characteristics of the data"""
    print("\n📊 DATA QUALITY ANALYSIS")
    print("-" * 30)
    
    try:
        # Original data
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv', comment='#')
        
        print(f"📈 Original Data:")
        print(f"   Training: {train_df.shape[0]} samples, {train_df.shape[1]} features")
        print(f"   Test: {test_df.shape[0]} samples, {test_df.shape[1]} features")
        print(f"   Target range: {train_df['beats_per_minute'].min():.1f} - {train_df['beats_per_minute'].max():.1f} BPM")
        print(f"   Missing values: {train_df.isnull().sum().sum()}")
        
        # Feature correlations with target
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['id', 'beats_per_minute']]
        
        correlations = train_df[feature_cols].corrwith(train_df['beats_per_minute']).sort_values(key=abs, ascending=False)
        
        print(f"\n🔗 Top feature correlations with BPM:")
        for i, (feature, corr) in enumerate(correlations.head(3).items()):
            print(f"   {i+1}. {feature}: {corr:.3f}")
            
    except Exception as e:
        print(f"   ❌ Error analyzing data: {e}")

def compare_predictions():
    """Compare predictions from different approaches"""
    print("\n🔍 PREDICTION COMPARISON")
    print("-" * 30)
    
    predictions = {}
    
    # Load different prediction files
    files = [
        ('Main Pipeline', 'data/submission_final.csv'),
        ('Notebook', 'data/submission.csv'),
    ]
    
    for name, filepath in files:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            predictions[name] = df['BeatsPerMinute'].values
            print(f"📋 {name}:")
            print(f"   Mean: {df['BeatsPerMinute'].mean():.2f}")
            print(f"   Std:  {df['BeatsPerMinute'].std():.2f}")
            print(f"   Range: {df['BeatsPerMinute'].min():.2f} - {df['BeatsPerMinute'].max():.2f}")
    
    # Compare predictions if we have multiple
    if len(predictions) > 1:
        print(f"\n📊 Prediction Agreement:")
        pred_names = list(predictions.keys())
        for i in range(len(pred_names)):
            for j in range(i+1, len(pred_names)):
                name1, name2 = pred_names[i], pred_names[j]
                corr = np.corrcoef(predictions[name1], predictions[name2])[0,1]
                mae = np.mean(np.abs(predictions[name1] - predictions[name2]))
                print(f"   {name1} vs {name2}: correlation={corr:.3f}, MAE={mae:.2f}")

def generate_summary_report(results):
    """Generate a comprehensive summary report"""
    print("\n📄 PROJECT SUMMARY REPORT")
    print("=" * 60)
    
    print(f"🕐 Execution Time: {results['timestamp']}")
    print(f"🎯 Project: BPM Prediction for Music Tracks")
    
    print(f"\n📦 Components Status:")
    for component, info in results['components'].items():
        status_emoji = "✅" if info['status'] == 'SUCCESS' else "❌" if info['status'] == 'FAILED' else "⚠️"
        print(f"   {status_emoji} {component.replace('_', ' ').title()}: {info['status']}")
    
    print(f"\n📁 Generated Files:")
    files_to_check = [
        'data/submission_final.csv',
        'data/submission.csv', 
        'data/train_experimental.csv',
        'data/test_experimental.csv'
    ]
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   ✅ {filepath} ({size} bytes)")
        else:
            print(f"   ❌ {filepath} (not found)")
    
    # Model performance summary
    if 'main_pipeline' in results['components'] and results['components']['main_pipeline']['status'] == 'SUCCESS':
        stats = results['components']['main_pipeline']['prediction_stats']
        print(f"\n🎯 Final Model Predictions:")
        print(f"   Mean BPM: {stats['mean']:.2f}")
        print(f"   Std Dev: {stats['std']:.2f}")
        print(f"   Range: {stats['min']:.2f} - {stats['max']:.2f}")
    
    print(f"\n🚀 Project Execution Complete!")
    print(f"   All core components have been executed.")
    print(f"   Prediction files are ready for submission.")

def main():
    """Main execution function"""
    # Run all components
    results = run_all_components()
    
    # Analyze data
    analyze_data_quality()
    
    # Compare predictions
    compare_predictions()
    
    # Generate summary
    generate_summary_report(results)
    
    return results

if __name__ == "__main__":
    results = main()

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
TRAIN_FILE = 'data/raw/train.csv'
TEST_FILE = 'data/raw/test.csv'
OUTPUTS_DIR = 'data/processed'
EXPERIMENTAL_TRAIN = 'train_experimental.csv'
EXPERIMENTAL_TEST = 'test_experimental.csv'
TARGET_COLUMN = 'beats_per_minute'

# Column mappings
TEST_COLUMN_MAPPING = {
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

ALL_NUMERIC_FEATURES = [
    'rhythm_score', 'audio_loudness', 'vocal_content', 'acoustic_quality',
    'instrumental_score', 'live_performance_likelihood', 'mood_score', 'energy'
]

CONTENT_FEATURES = ['vocal_content', 'acoustic_quality', 'instrumental_score']

def try_outlier_based_features(train_df, test_df):
    """Use outlier detection to create features"""
    print("🔍 Creating outlier-based features...")
    
    train_out = train_df.copy()
    test_out = test_df.copy()
    feature_cols = ALL_NUMERIC_FEATURES
    
    # Isolation Forest for outlier detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(train_df[feature_cols])
    
    # Apply to training data
    train_outlier_scores = iso_forest.decision_function(train_df[feature_cols])
    train_out['Outlier_Score'] = train_outlier_scores
    train_out['Is_Outlier'] = (train_outlier_scores < 0).astype(int)
    
    # Apply to test data
    test_outlier_scores = iso_forest.decision_function(test_df[feature_cols])
    test_out['Outlier_Score'] = test_outlier_scores
    test_out['Is_Outlier'] = (test_outlier_scores < 0).astype(int)
    
    # Local outlier factor (fit on training data)
    lof = LocalOutlierFactor(n_neighbors=min(20, len(train_df)-1), contamination=0.1, novelty=True)
    lof.fit(train_df[feature_cols])
    
    # Apply to training data
    train_lof_scores = lof.decision_function(train_df[feature_cols])
    train_out['LOF_Score'] = train_lof_scores
    train_out['Is_LOF_Outlier'] = (train_lof_scores < 0).astype(int)
    
    # Apply to test data
    test_lof_scores = lof.decision_function(test_df[feature_cols])
    test_out['LOF_Score'] = test_lof_scores
    test_out['Is_LOF_Outlier'] = (test_lof_scores < 0).astype(int)
    
    print(f"   ✅ Train outliers detected: {train_out['Is_Outlier'].sum()} / {len(train_out)}")
    print(f"   ✅ Test outliers detected: {test_out['Is_Outlier'].sum()} / {len(test_out)}")
    return train_out, test_out

def try_dimensionality_reduction_features(train_df, test_df):
    """Create features using dimensionality reduction"""
    print("📐 Creating dimensionality reduction features...")
    
    train_dim = train_df.copy()
    test_dim = test_df.copy()
    feature_cols = ALL_NUMERIC_FEATURES
    
    # Standardize for PCA/ICA
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # PCA components
    pca = PCA(n_components=5, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    for i in range(5):
        train_dim[f'PCA_Component_{i}'] = train_pca[:, i]
        test_dim[f'PCA_Component_{i}'] = test_pca[:, i]
    
    print(f"   ✅ PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # ICA components
    try:
        ica = FastICA(n_components=3, random_state=42, max_iter=1000)
        train_ica = ica.fit_transform(train_scaled)
        test_ica = ica.transform(test_scaled)
        
        for i in range(3):
            train_dim[f'ICA_Component_{i}'] = train_ica[:, i]
            test_dim[f'ICA_Component_{i}'] = test_ica[:, i]
        print(f"   ✅ ICA components created successfully")
    except:
        print(f"   ⚠️  ICA failed, skipping...")
    
    return train_dim, test_dim

def try_density_based_clustering(train_df, test_df):
    """Try K-means for finding patterns"""
    print("🎯 Creating clustering-based features...")
    
    train_cluster = train_df.copy()
    test_cluster = test_df.copy()
    feature_cols = ALL_NUMERIC_FEATURES
    
    # Standardize for clustering
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=min(8, len(train_df)), random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(train_scaled)
    test_clusters = kmeans.predict(test_scaled)
    
    train_cluster['KMeans_Cluster'] = train_clusters
    test_cluster['KMeans_Cluster'] = test_clusters
    
    # Distance to cluster centers
    train_distances = kmeans.transform(train_scaled)
    test_distances = kmeans.transform(test_scaled)
    
    train_cluster['Distance_to_Cluster'] = train_distances.min(axis=1)
    train_cluster['Avg_Distance_to_Clusters'] = train_distances.mean(axis=1)
    test_cluster['Distance_to_Cluster'] = test_distances.min(axis=1)
    test_cluster['Avg_Distance_to_Clusters'] = test_distances.mean(axis=1)
    
    # Local density using k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(20, len(train_df)))
    nn.fit(train_scaled)
    
    train_nn_distances, _ = nn.kneighbors(train_scaled)
    test_nn_distances, _ = nn.kneighbors(test_scaled)
    
    train_cluster['Local_Density'] = 1 / (train_nn_distances.mean(axis=1) + 1e-6)
    train_cluster['NN_Distance_Std'] = train_nn_distances.std(axis=1)
    test_cluster['Local_Density'] = 1 / (test_nn_distances.mean(axis=1) + 1e-6)
    test_cluster['NN_Distance_Std'] = test_nn_distances.std(axis=1)
    
    print(f"   ✅ Created {len(np.unique(train_clusters))} clusters")
    return train_cluster, test_cluster

def try_quantile_transformation(train_df, test_df):
    """Use quantile transformation for non-normal features"""
    print("📊 Applying quantile transformations...")
    
    train_quant = train_df.copy()
    test_quant = test_df.copy()
    
    # Apply quantile transformation to skewed features
    skewed_features = CONTENT_FEATURES + ['energy', 'rhythm_score', 'mood_score']
    
    qt_normal = QuantileTransformer(output_distribution='normal', random_state=42)
    qt_uniform = QuantileTransformer(output_distribution='uniform', random_state=42)
    
    for feature in skewed_features:
        # Normal transformation
        qt_normal.fit(train_df[[feature]])
        train_transformed_normal = qt_normal.transform(train_df[[feature]])
        test_transformed_normal = qt_normal.transform(test_df[[feature]])
        train_quant[f'{feature}_qtrans_normal'] = train_transformed_normal.flatten()
        test_quant[f'{feature}_qtrans_normal'] = test_transformed_normal.flatten()
        
        # Uniform transformation
        qt_uniform.fit(train_df[[feature]])
        train_transformed_uniform = qt_uniform.transform(train_df[[feature]])
        test_transformed_uniform = qt_uniform.transform(test_df[[feature]])
        train_quant[f'{feature}_qtrans_uniform'] = train_transformed_uniform.flatten()
        test_quant[f'{feature}_qtrans_uniform'] = test_transformed_uniform.flatten()
    
    print(f"   ✅ Applied quantile transformation to {len(skewed_features)} features")
    return train_quant, test_quant

def try_feature_interactions_advanced(train_df, test_df):
    """Create more sophisticated feature interactions"""
    print("🔗 Creating advanced feature interactions...")
    
    train_inter = train_df.copy()
    test_inter = test_df.copy()
    
    for df_inter in [train_inter, test_inter]:
        # Ratio-based features
        df_inter['Vocal_Instrumental_Ratio'] = df_inter['vocal_content'] / (df_inter['instrumental_score'] + 1e-6)
        df_inter['Acoustic_Energy_Ratio'] = df_inter['acoustic_quality'] / (df_inter['energy'] + 1e-6)
        df_inter['Live_Mood_Ratio'] = df_inter['live_performance_likelihood'] / (df_inter['mood_score'] + 1e-6)
        
        # Multiplicative interactions
        df_inter['Energy_Rhythm_Mood'] = df_inter['energy'] * df_inter['rhythm_score'] * df_inter['mood_score']
        df_inter['Vocal_Acoustic_Live'] = df_inter['vocal_content'] * df_inter['acoustic_quality'] * df_inter['live_performance_likelihood']
        
        # Duration-based interactions
        duration_minutes = df_inter['track_duration_ms'] / 60000
        df_inter['Energy_per_Minute'] = df_inter['energy'] / (duration_minutes + 1e-6)
        df_inter['Rhythm_per_Minute'] = df_inter['rhythm_score'] / (duration_minutes + 1e-6)
        df_inter['Mood_Duration_Product'] = df_inter['mood_score'] * duration_minutes
        
        # Weighted combinations
        df_inter['Danceability_Score'] = (0.4 * df_inter['energy'] + 
                                         0.3 * df_inter['rhythm_score'] + 
                                         0.3 * (-df_inter['audio_loudness'] / 20))
        
        df_inter['Relaxation_Score'] = (0.4 * df_inter['acoustic_quality'] + 
                                       0.3 * (1 - df_inter['energy']) +
                                       0.3 * df_inter['mood_score'])
    
    print(f"   ✅ Created advanced interaction features")
    return train_inter, test_inter

def analyze_target_distribution(df, target_col=TARGET_COLUMN):
    """Analyze the distribution of BPM values"""
    print("📈 Analyzing BPM distribution...")
    
    if target_col not in df.columns:
        return
    
    bpm = df[target_col]
    
    print(f"   📊 BPM Statistics:")
    print(f"      Mean: {bpm.mean():.2f}")
    print(f"      Median: {bpm.median():.2f}")
    print(f"      Std: {bpm.std():.2f}")
    print(f"      Min: {bpm.min():.2f}")
    print(f"      Max: {bpm.max():.2f}")
    print(f"      Skewness: {bpm.skew():.3f}")
    print(f"      Kurtosis: {bpm.kurtosis():.3f}")
    
    # Common BPM ranges
    ranges = [
        ('Very Slow', 0, 60),
        ('Slow', 60, 90),
        ('Moderate', 90, 120),
        ('Fast', 120, 150),
        ('Very Fast', 150, 200),
        ('Extreme', 200, 300)
    ]
    
    print(f"   🎵 BPM Range Distribution:")
    for name, low, high in ranges:
        count = ((bpm >= low) & (bpm < high)).sum()
        pct = count / len(bpm) * 100
        print(f"      {name:12s} ({low:3d}-{high:3d}): {count:6d} ({pct:5.1f}%)")

def run_experimental_pipeline():
    """Run all experimental approaches"""
    print("🧪 EXPERIMENTAL APPROACHES PIPELINE")
    print("=" * 50)
    
    # Find data files
    train_file = TRAIN_FILE
    test_file = TEST_FILE
    
    if not os.path.exists(train_file):
        print(f"❌ Could not find train.csv file at {train_file}!")
        return None, None
    
    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file, comment='#')
    
    # Rename test columns to match train format
    test_df = test_df.rename(columns=TEST_COLUMN_MAPPING)
    
    print(f"📊 Original training shape: {train_df.shape}")
    print(f"📊 Original test shape: {test_df.shape}")
    
    # Analyze target distribution
    analyze_target_distribution(train_df)
    
    print(f"\n🔬 Applying experimental feature engineering...")
    
    # 1. Outlier-based features
    train_exp, test_exp = try_outlier_based_features(train_df, test_df)
    
    # 2. Dimensionality reduction features
    train_exp, test_exp = try_dimensionality_reduction_features(train_exp, test_exp)
    
    # 3. Clustering-based features
    train_exp, test_exp = try_density_based_clustering(train_exp, test_exp)
    
    # 4. Quantile transformations
    train_exp, test_exp = try_quantile_transformation(train_exp, test_exp)
    
    # 5. Advanced interactions
    train_exp, test_exp = try_feature_interactions_advanced(train_exp, test_exp)
    
    print(f"\n✅ Experimental feature engineering complete!")
    print(f"📊 Final training shape: {train_exp.shape}")
    print(f"📊 Final test shape: {test_exp.shape}")
    print(f"🎯 Created {train_exp.shape[1] - train_df.shape[1]} new features")
    
    # Save experimental datasets
    train_exp_file = f'{OUTPUTS_DIR}/{EXPERIMENTAL_TRAIN}'
    test_exp_file = f'{OUTPUTS_DIR}/{EXPERIMENTAL_TEST}'
    
    train_exp.to_csv(train_exp_file, index=False)
    test_exp.to_csv(test_exp_file, index=False)
    
    print(f"💾 Experimental datasets saved:")
    print(f"   Training: {train_exp_file}")
    print(f"   Test: {test_exp_file}")
    
    return train_exp, test_exp

if __name__ == "__main__":
    train_exp, test_exp = run_experimental_pipeline()
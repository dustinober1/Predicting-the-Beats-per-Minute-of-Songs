def engineer_features(df, scaler=None, kmeans=None):
    """Apply feature engineering to any dataset"""
    df_fe = df.copy()
    
    # 1. Ratio and interaction features
    df_fe['Energy_to_Rhythm_Ratio'] = df_fe['Energy'] / (df_fe['RhythmScore'] + 1e-6)
    df_fe['Energy_to_Mood_Ratio'] = df_fe['Energy'] / (df_fe['MoodScore'] + 1e-6)
    df_fe['Acoustic_Energy_Product'] = df_fe['AcousticQuality'] * df_fe['Energy']
    df_fe['Vocal_Instrumental_Ratio'] = df_fe['VocalContent'] / (df_fe['InstrumentalScore'] + 1e-6)
    df_fe['Live_Energy_Interaction'] = df_fe['LivePerformanceLikelihood'] * df_fe['Energy']
    df_fe['Loudness_Energy_Product'] = np.abs(df_fe['AudioLoudness']) * df_fe['Energy']
    df_fe['Duration_Minutes'] = df_fe['TrackDurationMs'] / 60000
    df_fe['Energy_per_Minute'] = df_fe['Energy'] / df_fe['Duration_Minutes']
    df_fe['Rhythm_Duration_Product'] = df_fe['RhythmScore'] * df_fe['Duration_Minutes']
    df_fe['Audio_Complexity'] = (df_fe['AcousticQuality'] + df_fe['InstrumentalScore'] + df_fe['VocalContent']) / 3
    df_fe['Performance_Index'] = (df_fe['Energy'] + df_fe['RhythmScore'] + df_fe['MoodScore']) / 3
    
    # 2. Non-linear transformations
    original_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
                        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'Energy']
    
    # Polynomial features
    for feature in original_features:
        df_fe[f'{feature}_squared'] = df_fe[feature] ** 2
    
    # Log transformations
    skewed_features = ['VocalContent', 'AcousticQuality', 'InstrumentalScore']
    for feature in skewed_features:
        df_fe[f'{feature}_log'] = np.log(df_fe[feature] + 1e-6)
    
    # Square root transformations
    positive_features = ['RhythmScore', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 
                        'LivePerformanceLikelihood', 'MoodScore', 'Energy']
    for feature in positive_features:
        df_fe[f'{feature}_sqrt'] = np.sqrt(df_fe[feature])
    
    # Exponential features
    bounded_features = ['RhythmScore', 'VocalContent', 'AcousticQuality', 'Energy']
    for feature in bounded_features:
        df_fe[f'{feature}_exp'] = np.exp(df_fe[feature]) - 1
    
    # Inverse features
    for feature in original_features:
        if feature != 'AudioLoudness':
            df_fe[f'{feature}_inverse'] = 1 / (df_fe[feature] + 1e-6)
    
    # 3. Clustering features (only if scaler and kmeans are provided)
    if scaler is not None and kmeans is not None:
        clustering_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
                              'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'Energy']
        X_scaled = scaler.transform(df_fe[clustering_features])
        clusters = kmeans.predict(X_scaled)
        
        df_fe['Cluster'] = clusters
        
        # Distance to cluster centers
        distances = kmeans.transform(X_scaled)
        for i in range(kmeans.n_clusters):
            df_fe[f'Distance_to_Cluster_{i}'] = distances[:, i]
        
        # Cluster statistics (using training data statistics)
        cluster_bpm_means = {0: 118.93, 1: 119.11, 2: 118.93, 3: 119.20}  # From our analysis
        df_fe['Cluster_BPM_Mean'] = df_fe['Cluster'].map(cluster_bpm_means)
    
    return df_fe
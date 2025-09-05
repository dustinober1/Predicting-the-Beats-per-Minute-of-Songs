# Column mappings for data standardization

# Test dataset column mapping to training format
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

# Feature groups for different types of engineering
AUDIO_FEATURES = [
    'audio_loudness', 'energy', 'acoustic_quality'
]

RHYTHMIC_FEATURES = [
    'rhythm_score', 'mood_score'
]

CONTENT_FEATURES = [
    'vocal_content', 'instrumental_score', 'live_performance_likelihood'
]

TEMPORAL_FEATURES = [
    'track_duration_ms'
]

ALL_NUMERIC_FEATURES = AUDIO_FEATURES + RHYTHMIC_FEATURES + CONTENT_FEATURES + TEMPORAL_FEATURES

# Non-feature columns to exclude from modeling
NON_FEATURE_COLUMNS = [
    'id', 'title', 'artist', 'album', 'beats_per_minute'
]

# Target column
TARGET_COLUMN = 'beats_per_minute'

from .temporal_smoothing import TemporalMaskRefiner

# Create an alias for consistency with the original code
Temporal_Smoothing = type('Temporal_Smoothing', (), {
    'TemporalMaskRefiner': TemporalMaskRefiner
})()
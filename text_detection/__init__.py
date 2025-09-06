from .detect_subtitles_with_paddleocr import detect_subtitle_mask

# Create an alias for consistency with the original code
Detect_Subtitles_with_PaddleOCR = type('Detect_Subtitles_with_PaddleOCR', (), {
    'detect_subtitle_mask': staticmethod(detect_subtitle_mask)
})()
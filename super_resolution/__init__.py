from .upscale import upscale_frame

# Create an alias for consistency with the original code
Upscale = type('Upscale', (), {
    'upscale_frame': staticmethod(upscale_frame)
})()
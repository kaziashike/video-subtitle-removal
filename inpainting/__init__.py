from .inpaint_with_lama import inpaint_frame

# Create an alias for consistency with the original code
Inpaint_with_LaMa = type('Inpaint_with_LaMa', (), {
    'inpaint_frame': staticmethod(inpaint_frame)
})()
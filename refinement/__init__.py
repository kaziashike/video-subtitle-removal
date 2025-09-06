from .edge_feathering import feather_mask

# Create an alias for consistency with the original code
Edge_Feathering = type('Edge_Feathering', (), {
    'feather_mask': staticmethod(feather_mask)
})()
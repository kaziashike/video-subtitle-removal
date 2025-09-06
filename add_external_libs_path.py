import sys
import os

# Add the external_libs directory to the Python path
external_libs_path = os.path.join(os.path.dirname(__file__), 'external_libs')
sys.path.append(external_libs_path)

print(f"Added {external_libs_path} to Python path")
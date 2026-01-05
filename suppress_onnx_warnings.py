import os
import warnings

# Suppress ONNX Runtime warnings
os.environ['ORT_LOGGING_LEVEL'] = '3'  # Only show errors
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')

# Also suppress TurboMind warnings
os.environ['LMDEPLOY_LOG_LEVEL'] = 'ERROR'

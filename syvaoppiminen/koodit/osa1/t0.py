import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

print(tf.__version__) # Pitäisi tulostaa tensorflow 2.0 tai uudempi versio

# only when using gpu version
#tf.test.is_gpu_available(cuda_only=True)
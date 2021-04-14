import numpy as np
import utils
def gen_data():
        return {"x": np.random.random(size=(128, 32)).astype('float32'),
                "y": np.random.randint(2, size=(128, 1)).astype('int64')}

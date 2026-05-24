#log transform
#outlier handling
#feature creation

import numpy as np


def log_transform(df, columns):

    for col in columns:
        df[col] = np.log1p(df[col])

    return df
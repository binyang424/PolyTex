import numpy as np
import pandas as pd
import polykriging as pk

label_row = pd.date_range("20130101", periods=6, freq="D", tz="UTC")
label_col = list("ABCD")
data = np.random.randn(6, 4)
df = pd.DataFrame(data, index=label_row, columns=label_col)

# save
pk.pk_save("test.coo", df)

# load
df = pk.pk_load("test.coo")
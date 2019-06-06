import numpy as np
import pandas as pd

N = 5000

x = np.random.uniform(size=N)
y = np.random.uniform(size=N)
z = np.random.uniform(size=N)

xPy = x + y

x2 = 2*x
y2 = 2*y
z2 = 2*z

xSquared = x**2
ySquared = y**2
zSquared = z**2

df = pd.DataFrame.from_dict(
	{"x":x, "y":y, "z":z,
	"x2":x2, "y2":y2, "z2":z2,
	"xSquared":xSquared, "ySquared":ySquared, "zSquared":zSquared,
	"xPy_Label":xPy})

feature_names = ["x","x2","xSquared","y","y2","ySquared","z","z2","zSquared"]
df = df[feature_names]

df.to_csv("sum_synthetic.csv", index=False)











from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv(rf'Iranian_ml//norm_cfs/test.csv', sep=',')
# plt.plot(df['x'])
# plt.show()
folder = Path('F:\Iranian sandstones\correlation functions and permeability tensors\cfs')
df = pd.read_csv(r'F:\\Iranian sandstones\\correlation functions and permeability tensors\\cfs\\14_c2.csv', sep=',', header=None)
i = 0
for src_path in folder.glob('**/*'):
    df = pd.read_csv(src_path, sep=',', header=None)
    print(df.columns.size == 351)
    i = i+1
print(i)

# df = pd.read_csv(rf'Iranian_ml/big_14_c2.csv', sep=',', header=None).transpose()
# print(df)
# a = df[0]
# b = a.drop(index=0)
# c = b[1].replace('[', '')
# d = b[700].replace(']', '')
# b[1] = float(c)
# b[700] = float(d)

# a = df[1]
# f = a.drop(index=0)
# c = f[1].replace('[', '')
# d = f[700].replace(']', '')
# f[1] = float(c)
# f[700] = float(d)

# a = df[2]
# g = a.drop(index=0)
# c = g[1].replace('[', '')
# d = g[700].replace(']', '')
# g[1] = float(c)
# g[700] = float(d)

# result = np.array([b, f, g])

# a = pd.DataFrame(result).transpose()

# plt.plot(b)
# plt.plot(f)
# plt.plot(g)
# plt.show()
# # a.to_csv(rf'Iranian_ml/norm_cfs/test.csv', sep=',', header=['x', 'y', 'z'])
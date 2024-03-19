from pathlib import Path
import pandas as pd

filepath = {}
cfs_default = Path('D:\Work\KT\Iranian_ml\cfs\cfs_default')
cfs_norm = Path('D:\Work\KT\Iranian_ml\cfs\cfs_normalized')

for src_path in cfs_default.glob('**/*'):
        filepath.update({src_path.name: src_path})

for name, path in filepath.items():
        print(name)
        data = pd.read_csv(filepath[name], header=None, sep=',').transpose().iloc[1:]
        x = data[0]
        x.iloc[0] = float(x.iloc[0].split('[')[1])
        y = data[1]
        y.iloc[0] = float(y.iloc[0].split('[')[1])
        z = data[2]
        z.iloc[0] = float(z.iloc[0].split('[')[1])

        x.iloc[349] = float(x.iloc[349].split(']')[0])
        y.iloc[349] = float(y.iloc[349].split(']')[0])
        z.iloc[349] = float(z.iloc[349].split(']')[0])

        result = pd.DataFrame([x,y,z]).transpose()

        result.to_csv(cfs_norm / name, header=None, index=False)
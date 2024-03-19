from pathlib import Path
import pandas as pd

src = Path('D:\Work\KT\Iranian_ml\perm\default')
dst =  Path('D:\Work\KT\Iranian_ml\perm\\normalized')


names = ['14','16','22','24','30','34',]

x = pd.read_csv(src/ '14_x.dat').columns
y = pd.read_csv(src/ '14_y.dat').columns
z = pd.read_csv(src/ '14_z.dat').columns


for name in names:
    name_x  = name+'_x.dat'
    name_y  = name+'_y.dat'
    name_z  = name+'_z.dat'
    res_name = name + '.csv'
    match name:
        case '14':
            x = pd.read_csv(src/ name_x).columns
            y = pd.read_csv(src/ name_y).columns
            z = pd.read_csv(src/ name_z).columns
            res = pd.DataFrame([x,y,z])
            res.to_csv(dst / res_name, index=False, header=None)
        case '16':
            x = pd.read_csv(src/ name_x).columns
            y = pd.read_csv(src/ name_y).columns
            z = pd.read_csv(src/ name_z).columns
            res = pd.DataFrame([x,y,z])
            res.to_csv(dst / res_name, index=False, header=None)
        case '22':
            x = pd.read_csv(src/ name_x).columns
            y = pd.read_csv(src/ name_y).columns
            z = pd.read_csv(src/ name_z).columns
            res = pd.DataFrame([x,y,z])
            res.to_csv(dst / res_name, index=False, header=None)
        case '24':
            x = pd.read_csv(src/ name_x).columns
            y = pd.read_csv(src/ name_y).columns
            z = pd.read_csv(src/ name_z).columns
            res = pd.DataFrame([x,y,z])
            res.to_csv(dst / res_name, index=False, header=None)
        case '30':
            x = pd.read_csv(src/ name_x).columns
            y = pd.read_csv(src/ name_y).columns
            z = pd.read_csv(src/ name_z).columns
            res = pd.DataFrame([x,y,z])
            res.to_csv(dst / res_name, index=False, header=None)
        case '34':
            x = pd.read_csv(src/ name_x).columns
            y = pd.read_csv(src/ name_y).columns
            z = pd.read_csv(src/ name_z).columns
            res = pd.DataFrame([x,y,z])
            res.to_csv(dst / res_name, index=False, header=None)
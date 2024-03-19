import numpy as np
from skimage.transform import downscale_local_mean


dir = rf'F:\Iranian sandstones\sandstone_REV\1400\original'
outdir = rf'F:\Iranian sandstones\sandstone_REV\700\original'
#filename = rf'\16.raw'

filenames = [rf'\22.raw', rf'\24.raw', rf'\30.raw', rf'\34.raw']

for filename in filenames:
    stack = np.fromfile(dir+filename, dtype=np.uint8)

    stack = stack.reshape(1400, 1400, 1400)

    a = downscale_local_mean(stack, (2,2,2))

    a.astype('uint8').tofile(outdir+filename)
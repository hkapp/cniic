import numpy as np
import imageio
import math

for i in range(100):
  imgid = 801+i
  face = imageio.v3.imread('DIV2K_valid_HR/0' + str(imgid) + '.png')

  # Print metadata:
  # print(type(face))
  # print(face.shape, face.dtype)

  # Compute the color cube:
  print(np.min(face, axis=(0,1)))
  print(np.max(face, axis=(0,1)))

  # Compute the total number of colors used:
  flt = face.reshape(-1, face.shape[-1])
  print(flt.shape)
  ndistinct = np.unique(flt, axis=0).shape[0]
  print(ndistinct)
  print(math.log2(ndistinct))
  print(ndistinct*24 + face.shape[0]*face.shape[1]*math.log2(ndistinct))
  print(face.shape[0]*face.shape[1]*24)

# for i in range(face.shape[0]):
  # for j in range(face.shape[1]):
    # for l in range(face.shape[2]):
      # if face[i][j][l] == 255:
        # print(face[i][j])

import numpy as np
import imageio
import math
from queue import PriorityQueue

# https://en.wikipedia.org/wiki/Huffman_coding

# def pack_color(arr):
  # return (((arr[0] << 8) | arr[1]) << 8) | arr[2]

class ComparableMixin:
  def __eq__(self, other):
    return not self<other and not other<self
  def __ne__(self, other):
    return self<other or other<self
  def __gt__(self, other):
    return other<self
  def __ge__(self, other):
    return not self<other
  def __le__(self, other):
    return not other<self

class HufNode(ComparableMixin):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def __lt__(self, other):
    return id(self) < id(other)

  def is_leaf(self):
    return (self.right is None)

  def get_leaf(self):
    return self.left

  def get_pair(self):
    return (self.left, self.right)

def build_enc_tab(tree, root_dig, nbits, enc_tab):
  if tree.is_leaf():
    leaf = tree.get_leaf()
    print(type(leaf))
    enc_tab[leaf] = (root_dig, nbits)
  else:
    (left, right) = tree.get_pair()
    build_enc_tab(left, root_dig << 1, nbits + 1, enc_tab)
    build_enc_tab(right, (root_dig << 1) | 1, nbits + 1, enc_tab)

def prepare(data):
  # Compute the total number of colors used:
  flt = picraw.reshape(-1, picraw.shape[-1])
  print('- - - - - - - - - - -')
  print('Shape of the flattened image: ' + str(flt.shape))
  (uniq_colors, color_counts) = np.unique(flt, axis=0, return_counts=True)
  ndistinct = uniq_colors.shape[0]
  print('Number of distinct colors:' + str(ndistinct))
  # for shannon encoding:
  # print(math.log2(ndistinct))
  # print(ndistinct*24 + picraw.shape[0]*picraw.shape[1]*math.log2(ndistinct))
  print('Uncompressed size of the image: ' + str(picraw.shape[0]*picraw.shape[1]*24) + 'b')

  # print('First color: ' + str(uniq_colors[0]) + ' ~ ' + str(pack_color(uniq_colors[0])))
  print('First color: ' + str(uniq_colors[0]))
  print('Result shapes: ' + str(uniq_colors.shape) + ', ' + str(color_counts.shape))
  q = PriorityQueue()
  # Initialize the queue
  for i in range(uniq_colors.shape[0]):
    c = uniq_colors[i]
    n = color_counts[i]
    # Can't pass numpy arrays as right hand side value coz' Python
    # https://stackoverflow.com/questions/42236820/adding-numpy-array-to-a-heap-queue
    node = HufNode(c, None)
    q.put((n, node))

  print('Queue init done')

  # Repeatedly merge nodes
  while q.qsize() > 1:
    (n1, t1) = q.get()
    (n2, t2) = q.get()
    # print(type(t1))
    # print(type(t2))
    t3 = HufNode(t1, t2)
    # print(type((n1+n2, t3)))
    q.put((n1 + n2, t3))

  print('Merge phase done')

  (tot_occ, huf_tree) = q.get()
  print(tot_occ)

  # Recursively build the codes and encoding table
  res = dict()
  build_enc_tab(huf_tree, 0, 0, res)

  print(res)
  return res

for i in range(1):#range(100):
  imgid = 801+i
  picraw = imageio.v3.imread('DIV2K_valid_HR/0' + str(imgid) + '.png')

  prepare(picraw)

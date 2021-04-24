import numpy as np

def compute_ap1(pos, ranked):
  old_recall = 0.0
  old_precision = 1.0
  ap = 0.0
  intersect_size = 0
  k = 0
  for x in ranked:
    if x in pos: intersect_size +=1
    recall = intersect_size/float(len(pos))
    precision = intersect_size/(k+1.0)
    ap += (recall - old_recall)*((old_precision + precision)/2.0)
    old_recall = recall
    old_precision = precision
    k += 1
  return ap

def compute_map_and_print1(dataset, ranks, gnd):
  ap = 0.0
  for q in range(ranks.shape[1]):
    ap += compute_ap1(gnd[q]['ok'], ranks[:, q]) 
  mAP = ap/float(ranks.shape[1])
  print('>> '+dataset+' mAP:', np.around(mAP*100, decimals=2))
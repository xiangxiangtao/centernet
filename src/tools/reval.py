#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Xingyi Zhou
# --------------------------------------------------------

# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'voc_eval_lib'))

from src.tools.voc_eval_lib.model.test import apply_nms
from src.tools.voc_eval_lib.datasets.pascal_voc import pascal_voc,dataset_name,myclass,dataset_mode,img_ext
import pickle
import os, argparse
import numpy as np
import json



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Re-evaluate results')
  parser.add_argument('--detection_file', type=str)
  parser.add_argument('--output_dir', help='results directory', type=str)
  # parser.add_argument('--imdb', dest='imdb_name',
  #                     help='dataset to re-evaluate',
  #                     default='voc_2007_test', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to re-evaluate',
                      default=dataset_name, type=str)
  parser.add_argument('--matlab', dest='matlab_eval',
                      help='use matlab for evaluation',
                      action='store_true')
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                      action='store_true')
  parser.add_argument('--nms', dest='apply_nms', help='apply nms',
                      action='store_true')

  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)

  args = parser.parse_args()
  return args


def from_dets(imdb_name, detection_file, args,eval_split):
  print('from_dets...')
  imdb = pascal_voc(eval_split, '2012')
  imdb.competition_mode(args.comp_mode)
  imdb.config['matlab_eval'] = args.matlab_eval
  with open(os.path.join(detection_file), 'rb') as f:
    if 'json' in detection_file:
      dets = json.load(f)
    else:
      dets = pickle.load(f, encoding='latin1')
  # import pdb; pdb.set_trace()
  if args.apply_nms:
    print('Applying NMS to all detections')
    test_nms = 0.5
    nms_dets = apply_nms(dets, test_nms)
  else:
    nms_dets = dets
  # nms_dets=np.array(nms_dets)
  print('Evaluating detections')
  map=imdb.evaluate_detections(nms_dets)
  return map

def dets(eval_split,detection_file):
  print('dets...')
  args = parse_args()

  imdb_name = args.imdb_name
  map=from_dets(imdb_name, detection_file, args,eval_split)
  return map

if __name__ == '__main__':
  args = parse_args()

  imdb_name = args.imdb_name
  from_dets(imdb_name, args.detection_file, args)

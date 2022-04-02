from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch

import sys
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

from src.lib.opts import opts
# from src.lib.logger import Logger
from src.lib.utils.utils import AverageMeter
from src.lib.datasets.dataset_factory import dataset_factory
from src.lib.detectors.detector_factory import detector_factory
from src.tools.voc_eval_lib.datasets.pascal_voc import dataset_name
from src.tools.voc_eval_lib.model.config import cfg
import time



class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  # print(opt)
  # Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

def eval(opt,eval_split):
  print("eval...")
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  # print(opt)
  # print(opt.heads)
  # Logger(opt)
  Detector = detector_factory[opt.task]

  dataset = Dataset(opt, eval_split)

  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}

  # timers
  _t = {'im_detect': Timer(), 'misc': Timer()}

  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    _t['im_detect'].tic()

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    detect_time = _t['im_detect'].toc(average=True)
    print('im_detect: {:d}/{:d} {:.3f}s'.format(ind + 1,num_iters, detect_time))

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()

  results_json_dir=os.path.join(cfg.DATA_DIR,"cache","{}".format(dataset_name),"{}".format(eval_split),"result")
  if not os.path.exists(results_json_dir):
    os.makedirs(results_json_dir)
  map=dataset.run_eval(results, results_json_dir,eval_split)
  # print('*'*50)
  # print(map)
  return map

def eval_one_weight():
  eval_split="val"

  opt = opts().parse()
  # print(opt.heads)
  opt.load_model='models/weight_centernet_gas_composite18_1_epoch_15.pth'####################
  if opt.not_prefetch_test:
    eval(opt,eval_split)
  else:
    prefetch_test(opt)

def eval_weights_in_folder():
  eval_split="val"

  opt = opts().parse()
  # print(opt.heads)

  weight_folder=r"../exp/ctdet/dla34-mod/centernet_composite_18.1_gmy"
  weight_list=os.listdir(weight_folder)
  weight_list.remove("model_current.pth")
  weight_list.sort(key=lambda x:int(x[x.index("epoch")+6:x.index(".pth")]))
  for weight in weight_list:
    print("-"*100)
    print(weight)
    weight_path=os.path.join(weight_folder,weight)
    opt.load_model=weight_path

    if opt.not_prefetch_test:
      eval(opt,eval_split)
    else:
      prefetch_test(opt)

if __name__ == '__main__':
  eval_one_weight()

  # eval_weights_in_folder()
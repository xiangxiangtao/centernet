from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..'))

import src._init_paths

import torch
import torch.utils.data


from src.lib.opts import opts
from src.lib.models.model import create_model, load_model, save_model
from src.lib.models.data_parallel import DataParallel
# from src.lib.logger import Logger
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.trains.train_factory import train_factory
from src.logger import Logger
from src.test import *
from src.tools.voc_eval_lib.datasets.pascal_voc import dataset_name

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print('*'*50)
  print(opt)

  # logger = Logger(opt)
  logger = Logger("logs/gas_centernet")
  model_save_folder=os.path.join(opt.save_dir, "centernet_{}".format(dataset_name))#/home/ecust/txx/project/gmy_2080_copy/CenterNet-master/CenterNet-master/exp/ctdet/dla34-mod
  if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  print("train data")
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'),
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )
  print("val data")
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  print('Starting training...')
  best = 0#best map
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    print('*'*50)
    print('epoch={}'.format(epoch))
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    # logger.write('epoch: {} |'.format(epoch))
    # print('*'*20)
    print(log_dict_train)
    # print('*'*20)
    for k, v in log_dict_train.items():
      # logger.scalar_summary('train_{}'.format(k), v, epoch)
      # logger.write('{} {:8f} | '.format(k, v))

      # log
      if k!='time':
        logger.scalar_summary('train_'+k,v, epoch)


    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(model_save_folder, 'model_current.pth'),
                 epoch, model, optimizer)
      print("\n---- Evaluating Model ----")
      opt.load_model = os.path.join(model_save_folder, 'model_current.pth') ####################
      map=eval(opt,eval_split="val")

      # log
      logger.scalar_summary('val_map', map, epoch)



      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      # print('*' * 20)
      # print(log_dict_val)

      for k, v in log_dict_val.items():
        # logger.scalar_summary('val_{}'.format(k), v, epoch)
        # logger.write('{} {:8f} | '.format(k, v))

        # log
        if k != 'time':
          logger.scalar_summary('val_' + k, v, epoch)

      # if log_dict_val[opt.metric] < best:
      #   best = log_dict_val[opt.metric]
      #   save_model(os.path.join(opt.save_dir, 'model_best.pth'),
      #              epoch, model)
      # if map > best:
      #   print("save epoch {} model".format(epoch))
      #   best = map
      save_model(os.path.join(model_save_folder, 'centernet_epoch_{}.pth'.format(epoch)),
                   epoch, model)

    # else:
    #   save_model(os.path.join(opt.save_dir, 'model_last.pth'),
    #              epoch, model, optimizer)
    # logger.write('\n')
    if epoch in opt.lr_step:
      # save_model(os.path.join(model_save_folder, 'model_{}.pth'.format(epoch)),
      #            epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  # logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
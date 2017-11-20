import sys
import os.path
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

is_cuda = torch.cuda.is_available()
batch_size = 8
pin_memory = True if is_cuda else False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if is_cuda:
  cudnn.benchmark = True
lr = 1e-4

class SimpleNet():
  # TODO: remove this init
  def __init__(self):
    None
  
  def set_model(self, model):
    if is_cuda:
      self.model = model.cuda()
    else:
      self.model = model
    self.optimizer = optim.Adam(model.fc.parameters(), lr, weight_decay=1e-4)
    self.criterion = nn.CrossEntropyLoss().cuda() if is_cuda else nn.CrossEntropyLoss()
  
  def train(self, train_loader, epoch):
    train_model(train_loader, self.model, self.criterion, self.optimizer, epoch)
  
  def validate(self, val_loader, epoch):
    validate_model(val_loader, self.model, self.criterion, epoch)
    
class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def train_model(train_loader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  acc = AverageMeter()
  end = time.time()
    
  # switch to train mode
  model.train()
    
  for i, (images, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    target = target.cuda(async=True) if is_cuda else target
    image_var = torch.autograd.Variable(images)
    label_var = torch.autograd.Variable(target)
    if is_cuda:
      image_var = image_var.cuda()
      label_var = label_var.cuda()

    # compute y_pred
    y_pred = model(image_var)
    loss = criterion(y_pred, label_var)

    # measure accuracy and record loss
    prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
    losses.update(loss.data[0], images.size(0))
    acc.update(prec1[0], images.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

def validate_model(val_loader, model, criterion, epoch):
  batch_time = AverageMeter()
  losses = AverageMeter()
  acc = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  for i, (images, labels) in enumerate(val_loader):
    labels = labels.cuda(async=True) if is_cuda else labels
    image_var = torch.autograd.Variable(images, volatile=True)
    label_var = torch.autograd.Variable(labels, volatile=True)
    
    if is_cuda:
      image_var = image_var.cuda()
      label_var = label_var.cuda()

    # compute y_pred
    y_pred = model(image_var)
    loss = criterion(y_pred, label_var)

    # measure accuracy and record loss
    prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
    losses.update(loss.data[0], images.size(0))
    acc.update(prec1[0], images.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

  print('   * EPOCH {epoch} | Accuracy: {acc.avg:.3f} | Loss: {losses.avg:.3f}'.format(epoch=epoch,
                                                                                       acc=acc,
                                                                                       losses=losses))
  return acc.avg
    
def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  global lr
  lr = lr * (0.1**(epoch // 30))
  for param_group in optimizer.state_dict()['param_groups']:
    param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = y_actual.size(0)

  _, pred = y_pred.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))

  return res

def dataset(path, transformations):
  dataset = datasets.ImageFolder(path, transformations)
  return data.DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=pin_memory)

def training_data(traindir):
  return dataset(traindir, transforms.Compose([
      transforms.RandomSizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize
  ]))

def validation_data(valdir):
  return dataset(valdir, transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize
  ]))
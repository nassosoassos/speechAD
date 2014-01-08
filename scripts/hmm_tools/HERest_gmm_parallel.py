#!/usr/bin/python
# HERest -d tmp_htk/model -S tmp_htk/data.scp -L tmp_htk/data -p 7 tmp_htk/models.list 
import argparse
from multiprocessing import Process, cpu_count
import os
import itertools
import math
import subprocess

def grouper(n, iterable, fillvalue=None):
  #grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
  args = [iter(iterable)] * n
  return itertools.izip_longest(fillvalue=fillvalue, *args)

def herest_thread(model_dir, data_list, label_dir, model_list, proc_id):
  assert(proc_id>0)
  cmd = ['HERest','-d',model_dir,'-M', model_dir, '-S',data_list,'-L',label_dir,'-p', str(proc_id), model_list]
  #print cmd
  subprocess.call(cmd)

def herest_join(model_dir, acc_list, label_dir, model_list):
  cmd = ['HERest','-d',model_dir,'-M',model_dir,'-S',acc_list,'-L',label_dir,'-p', str(0), model_list]
  #print cmd
  subprocess.call(cmd)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run HERest for gmm training on multiple cores.')
  parser.add_argument('model_dir', metavar='model_dir', 
      type=str, help='directory where the gmm lies' )
  parser.add_argument('label_dir', metavar='label_dir', 
      type=str, help='directory where the labels lie')
  parser.add_argument('scp_file', metavar='scp_file', 
      type=str, help='list of the training files')
  parser.add_argument('n_iterations', metavar='n_iterations',
      type=int, help='number of EM iterations')
  parser.add_argument('model_list', metavar='model_list',
      type=str, help='list of models to be trained')
  parser.add_argument('--tmp_dir', metavar='tmp_dir',
      type=str, default='tmp', help='temporary directory')
  
  args = parser.parse_args()
  n_processors = cpu_count()

  if not os.path.exists(args.tmp_dir):
    os.makedirs(args.tmp_dir)

  # Read the list of data and split into multiple chunks
  scp_fid = open(args.scp_file,'r')
  train_files = scp_fid.readlines()
  n_train_files = len(train_files)

  scp_fid.close()

  lists_fids = []
  if n_processors==1:
    n_used_processors = 1
  else:
    n_used_processors = n_processors-1

  n_files_per_processor = int(math.ceil(float(n_train_files)/float(n_used_processors)))

  print args.n_iterations
  for it in range(0, args.n_iterations):
    file_sets = grouper(n_files_per_processor, train_files, '')
    procs = []
    for i_proc in range(1,n_used_processors+1):
      list_file = os.path.join(args.tmp_dir,'data'+str(i_proc)+'.scp')
      list_fid = open(list_file,'w')
      for f in file_sets.next():
        list_fid.write(f)
      list_fid.close()
      argums = (args.model_dir, list_file, args.label_dir, args.model_list, i_proc, )
      p = Process(target=herest_thread, args=argums)
      p.start()
      procs.append(p)
  
    acc_list = os.path.join(args.tmp_dir,'acc.list')
    acc_list_fid = open(acc_list,'w')
    for i_proc in range(1, n_used_processors+1):
      p = procs[i_proc-1]
      p.join()
      her_file = os.path.join(args.model_dir,'HER'+str(i_proc)+'.acc')
      acc_list_fid.write(her_file+'\n')

    acc_list_fid.close()

    herest_join(args.model_dir, acc_list, args.label_dir, args.model_list)










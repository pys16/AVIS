"""A program to apply different well-known deep learning architectures (AlexNet, ResNet, NiN, VGG, GoogLeNet, DenseNet) to ImageNet or any other large datasets. 

For more information on the features and how to use this code, please refer to the readme file.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np

import tensorflow as tf

from data_loader import loader
from architectures.model import model
from architectures.common import SAVE_VARIABLES
import sys
import argparse
import utils

"""
  This method trains a deep neural network using the provided configuration.
  For more details on the different steps of training, please read inline comments below.

  Args:
    sess: a tensorflow session to build and run the computational graph.
    args: augmented command line arguments determiing the details of training.
  Returns:
    nothing.
"""

"""
 This method evaluates (or just run) a trained deep neural network using the provided configuration.
 For more details on the different steps of training, please read inline comments below.

 Args:
     sess: a tensorflow session to build and run the computational graph.
     args: augmented command line arguments determiing the details of evaluation.
 Returns:
     nothing.
"""

def do_evaluate(sess, args):
  with tf.device('/cpu:0'):
    # Images and labels placeholders
    images_ph= tf.placeholder(tf.float32, shape=(None,)+ tuple(args.processed_size), name='input')
    labels_ph= tf.placeholder(tf.int32, shape=(None), name='label')

    # a placeholder for determining if we train or validate the network. This placeholder will be used to set dropout rates and batchnorm paramaters.
    is_training_ph= tf.placeholder(tf.bool, name='is_training')

    # build a deep learning model using the provided configuration
    dnn_model= model(images_ph, labels_ph, utils.loss, None, 0.0, args.architecture, args.num_classes, is_training_ph, args.transfer_mode)

    # creating an input pipeline to read data from disk
    # a placeholder for setting the input pipeline batch size. This is employed to ensure that we feed each validation example only once to the network.
    batch_size_tf= tf.placeholder_with_default(args.batch_size, shape=())

    # a data loader pipeline to read test data
    val_loader= loader(args.val_info, args.delimiter, args.raw_size, args.processed_size, False, batch_size_tf, args.num_prefetch, args.num_threads, 
            args.path_prefix, inference_only= args.inference_only)

    # if we want to do inference only (i.e. no label is provided) we only load images and their paths
    if not args.inference_only:
      val_images, val_labels, val_info = val_loader.load()
    else:
      val_images, val_info = val_loader.load()

    # get evaluation operations from the dnn model
    eval_ops = dnn_model.evaluate_ops(args.inference_only)

    # Build an initialization operation to run below.
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # Load pretrained parameters from disk
    dnn_model.load(sess, args.log_dir)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # evaluation 
    if not args.inference_only:
      total_loss = utils.AverageMeter() # Measures cross entropy loss
      top1 = utils.AverageMeter() # Measures top-1 accuracy
      topn = utils.AverageMeter() # Measures top-n accuracy

      # Open an output file to write predictions
      out_file = open(args.save_predictions,'w')
      predictions_format_str = ('%d, %s, %s, %s, %s\n')
      
      for step in range(args.num_val_batches):
        # Load a batch of data
        val_img, val_lbl, val_inf = sess.run([val_images, val_labels, val_info],feed_dict={batch_size_tf: args.num_val_samples%args.batch_size} if step==args.num_val_batches-1 else None)

        # Evaluate the network on the loaded batch
        val_loss, top1_predictions, topn_predictions, topnguesses, topnconf = sess.run(eval_ops, feed_dict={ images_ph: val_img, labels_ph: val_lbl, is_training_ph: False}, 
                options= args.run_options, run_metadata= args.run_metadata)

        current_batch_size= val_lbl.shape[0]
        total_loss.update(val_loss, current_batch_size)
        top1.update(top1_predictions, current_batch_size)
        topn.update(topn_predictions, current_batch_size)
        print('Batch Number: %d of %d, Top-1 Hit: %d, Top-%d Hit: %d, Loss %.2f, Top-1 Accuracy: %.2f, Top-%d Accuracy: %.2f'%
        (step, args.num_val_batches, top1.sum, args.top_n, topn.sum, total_loss.avg, top1.avg, args.top_n, topn.avg))

        # log results into an output file
        for i in range(0,val_inf.shape[0]):
          out_file.write(predictions_format_str%(step*args.batch_size+i+1, val_inf[i], val_loader.label_dict[val_lbl[i]],
          ', '.join('%d' % item for item in topnguesses[i]),
          ', '.join('%.4f' % item for item in topnconf[i])))
          out_file.flush()

      out_file.close()
    #inference
    else:
        
      # Open an output file to write predictions
      out_file = open(args.save_predictions,'w')
      predictions_format_str = ('%d, %s, %s, %s\n')

      for step in range(args.num_val_batches):
        # Load a batch of data
        val_img, val_inf = sess.run([val_images, val_info], feed_dict={batch_size_tf: args.num_val_samples%args.batch_size} if step==args.num_val_batches-1 else None)

        # Run the network on the loaded batch
        topnguesses, topnconf = sess.run(eval_ops, feed_dict={ images_ph: val_img, is_training_ph: False}, options= args.run_options, run_metadata= args.run_metadata)
        print('Batch Number: %d of %d is done'%(step, args.num_val_batches))

        # Log to an output file
        for i in range(0,val_inf.shape[0]):
          out_file.write(predictions_format_str%(step*args.batch_size+i+1, val_inf[i],
          ', '.join('%d' % item for item in topnguesses[i]),
          ', '.join('%.4f' % item for item in topnconf[i])))
          out_file.flush()

      out_file.close()

    coord.request_stop()
    coord.join(threads)
    sess.close()

def main(mode=1):  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(description='Process Command-line Arguments')
    # parser.add_argument('command',default='inference', action= 'store', help= 'Determines what to do: train, evaluate, or inference')
    parser.add_argument('--raw_size', nargs= 3, default= [256,256,3], type= int, action= 'store', help= 'The width, height and number of channels of images for loading from disk')
    parser.add_argument('--processed_size', nargs= 3, default= [224,224,3], type= int, action= 'store', help= 'The width and height of images after preprocessing')
    parser.add_argument('--batch_size', default= 8, type= int, action= 'store', help= 'The batch size for training, evaluating, or inference')
    parser.add_argument('--num_classes', default= 8 , type=int, action='store', help= 'The number of classes')
    parser.add_argument('--num_prefetch', default= 2000 , type= int, action= 'store', help= 'The number of pre-fetched images in the training queue, reduce this to consume less RAM')
    parser.add_argument('--num_epochs', default= 60, type= int, action= 'store', help= 'The number of training epochs')
    parser.add_argument('--path_prefix', default= '', action='store', help= 'the prefix address for images')
    parser.add_argument('--train_info', default= None, action= 'store', help= 'Name of the file containing addresses and labels of training images')
    parser.add_argument('--val_info', default= 'validation.txt', action= 'store', help= 'Name of the file containing addresses and labels of validation images')
    parser.add_argument('--shuffle', default= True, type= bool, action= 'store',help= 'Shuffle training data or not')
    parser.add_argument('--num_threads', default= 20, type= int, action='store', help= 'The number of threads for loading data')
    parser.add_argument('--log_dir', default= "./model/Img", action= 'store', help= 'Path for saving Tensorboard info and checkpoints')
    parser.add_argument('--snapshot_prefix', default= 'snapshot', action= 'store', help= 'Prefix for checkpoint files')
    parser.add_argument('--architecture', default= 'resnet18', help= 'The DNN architecture')
    parser.add_argument('--run_name', default= 'Run'+str(time.strftime("-%d-%m-%Y_%H-%M-%S")), action= 'store', help= 'Name of the experiment')
    parser.add_argument('--num_gpus', default= 1, type= int, action= 'store', help= 'Number of GPUs (Only for training)')
    parser.add_argument('--log_device_placement', default= False, type= bool, help= 'Whether to log device placement or not')
    parser.add_argument('--delimiter', default= ' ', action= 'store', help= 'Delimiter of the input files')
    parser.add_argument('--retrain_from', default= None, action= 'store', help= 'Continue Training from a snapshot file')
    parser.add_argument('--log_debug_info', default= False, action= 'store', help= 'Logging runtime and memory usage info')
    parser.add_argument('--num_batches', default= -1, type= int, action= 'store', help= 'The number of batches per epoch')
    parser.add_argument('--transfer_mode', default = [0], nargs='+', type= int, help= 'Transfer mode 0=None , 1=Tune last layer only , 2= Tune all the layers, 3= Tune the last layer at early epochs     (it could be specified with the second number of this argument) and then tune all the layers')
    parser.add_argument('--LR_policy', default='piecewise_linear', help='LR change policy type (piecewise_linear, constant, exponential)')
    parser.add_argument('--WD_policy', default='piecewise_linear', help='WD change policy type (piecewise_linear, constant, exponential)')
    parser.add_argument('--LR_details', default= '19, 30, 44, 53, 0.01, 0.005, 0.001, 0.0005, 0.0001', help='LR change details')
    parser.add_argument('--WD_details', default='30, 0.0005, 0.0', help='WD change details') 
    parser.add_argument('--optimizer', default= 'momentum', help= 'The optimization algorithm (SGD, Momentum, Adam, RMSprop, ...)')
    parser.add_argument('--top_n', default= 5, type= int, action= 'store', help= 'Specify the top-N accuracy')
    parser.add_argument('--max_to_keep', default= 5, type= int, action= 'store', help= 'Maximum number of snapshot files to keep')
    parser.add_argument('--save_predictions', default= 'preds.txt', action= 'store', help= 'Save top-n predictions of the networks along with their confidence in the specified file')
    args = parser.parse_args()

    if mode!=1:
        args.log_dir='./model/Sound'
        args.save_predictions='predsound.txt'
        args.val_info='validsound.txt'
        args.batch_size=16
        # args.raw_size=[256,256,1]

    # Spliting examples between different GPUs
    args.chunked_batch_size = int(args.batch_size/args.num_gpus)

    # Logging the runtime information if requested
    if args.log_debug_info:
      args.run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
      args.run_metadata = tf.RunMetadata()
    else:
      args.run_options = None
      args.run_metadata = None

    # Creating a session to run the built graph
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement= True, 
        log_device_placement= args.log_device_placement))

    print(args)
    """
    if args.command.lower()=='train':
      #Assert input args
      assert args.train_info is not None, "No training dataset is provided, please provide an input file using --train_info option"

      # Counting number of training examples
      if args.num_batches==-1:
        args.num_samples, args.num_batches= utils.count_input_records(args.train_info, args.batch_size)
      else:
        args.num_samples, _ = utils.count_input_records(args.train_info, args.batch_size)

      # Counting number of validation examples
      if args.val_info is not None:
        args.num_val_samples, args.num_val_batches= utils.count_input_records(args.val_info, args.batch_size)
        args.run_validation= True
      else:
        args.run_validation= False

      # creating the logging directory
      if args.log_dir is None:
        args.log_dir= args.architecture+"_"+args.run_name

      if tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
      tf.gfile.MakeDirs(args.log_dir)
      print("Saving everything in "+args.log_dir)

      # do training
      do_train(sess, args)

    elif args.command.lower()=='eval':
      # set config
      args.inference_only= False

      # Counting number of training examples
      assert args.val_info is not None, "No evaluation dataset is provided, please provide an input file using val_info option"
      args.num_val_samples, args.num_val_batches= utils.count_input_records(args.val_info, args.batch_size)
      
      # do evaluation
      do_evaluate(sess, args)

    elif args.command.lower()=='inference':
      # set config
      args.inference_only= True

      # Counting number of test examples
      assert args.val_info is not None, "No inference dataset is provided, please provide an input file using --val_info option"
      args.num_val_samples, args.num_val_batches= utils.count_input_records(args.val_info, args.batch_size)

      # do testing
      do_evaluate(sess, args)
    else:
      print("Command not found")
    """
    # set config
    args.inference_only= True

    # Counting number of test examples
    assert args.val_info is not None, "No inference dataset is provided, please provide an input file using --val_info option"
    args.num_val_samples, args.num_val_batches= utils.count_input_records(args.val_info, args.batch_size)
    # do testing
    do_evaluate(sess, args)
    sess.close()

if __name__ == '__main__':
  main()

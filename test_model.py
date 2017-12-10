#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocab_filepath", "runs/1512222837/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1512222837/checkpoints/model-5000", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("q1", "Question 1?", "Question 1 (Default: Question 1)")
tf.flags.DEFINE_string("q2", "Question 2?", "Question 2 (Default: Question 2)")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


if FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

q1 = FLAGS.q1
q2 = FLAGS.q2

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)

        ans_pred, ans_sim = sess.run([predictions, sim], {input_x1: [q1], input_x2: [q2], dropout_keep_prob: 1.0})

        print ans_pred, ans_sim


import tensorflow.contrib.keras as kr
import tensorflow as tf
import argparse
import numpy as np
import os
from sklearn import metrics
from utils.dataprocessor import *
#hyper parameters
parser = argparse.ArgumentParser(description='Text CNN model case test program')
parser.add_argument('--model', type=str, default='model/textcnn.pb', help='the path for the model')
parser.add_argument('--dictionary', type=str, default='dict/textcnn.dict', help='the path for dictionary')
parser.add_argument('--labels', type=str, default='labels/textcnn.labels', help='the path for labels')
parser.add_argument('--seq_length', type=int, default=50, help='the length of sequence for text padding')
parser.add_argument('--testdata', type=str, default='data/testdata', help='the path of test data')
parser.add_argument('--tensor_input', type=str, default='input_x:0', help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_dropout', type=str, default='keep_prob:0', help='the dropout op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_output', type=str, default='score/my_output:0', help='the output op_name for graph, format： <op_name>:<output_index>')
args_in_use = parser.parse_args()


with tf.Graph().as_default():

    id_to_cat, cat_to_id = read_labels(args_in_use.labels)
    word_to_id = read_dict(args_in_use.dictionary)
    contents, y_test_cls = process_file(args_in_use.testdata, word_to_id, args_in_use.seq_length, cat_to_id)

    output_graph_def = tf.GraphDef()
    with open(args_in_use.model, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_text = sess.graph.get_tensor_by_name(args_in_use.tensor_input)#"<op_name>:<output_index>"., must be 2d array otherwise it will not be used
        dropout = sess.graph.get_tensor_by_name(args_in_use.tensor_dropout)
        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)
        y_pred_cls = []
        for x,y in batch_itr(contents, y_test_cls, batch_size = 4):
            feed_dict = {
                test_text : x,
                dropout: 1.0
            }
            y_pred_cls.extend(np.argmax(sess.run(output, feed_dict=feed_dict), 1))



        print('=====testing=====')
        target_idx = set(list(set(y_test_cls))+list(set(y_pred_cls)))
        # map classification index into class name
        target_names = [id_to_cat.get(x) for x in target_idx]
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=target_names, digits=4))

import tensorflow as tf
import argparse
import numpy as np
import os
from utils.dataprocessor import *
#hyper parameters
parser = argparse.ArgumentParser(description='Text CNN model case test program, exit with q')
parser.add_argument('--model', type=str, default='model/textcnn.pb', help='the path for the model')
parser.add_argument('--dictionary', type=str, default='dict/textcnn.dict', help='the path for dictionary')
parser.add_argument('--labels', type=str, default='labels/textcnn.labels', help='the path for labels')
parser.add_argument('--seq_length', type=int, default=50, help='the length of sequence for text padding')
parser.add_argument('--tensor_input', type=str, default='input_x:0', help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_dropout', type=str, default='keep_prob:0', help='the dropout op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_output', type=str, default='score/my_output:0', help='the output op_name for graph, format： <op_name>:<output_index>')
args_in_use = parser.parse_args()



with tf.Graph().as_default():
    word_to_id = read_dict(args_in_use.dictionary)
    id_to_cat, _ = read_labels(args_in_use.labels)
    output_graph_def = tf.GraphDef()

    with open(args_in_use.model, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')# remember to set name='' otherwise it will cannot find operations in graph
        #name: (Optional.) A prefix that will be prepended to the names in graph_def. Note that this does not apply to imported function names. Defaults to "import".
# Just disables the warning, doesn't enable AVX/FMA

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_text = sess.graph.get_tensor_by_name(args_in_use.tensor_input)#"<op_name>:<output_index>".
        drop = sess.graph.get_tensor_by_name(args_in_use.tensor_dropout)
        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)

        while 1:
            sentence = input("enter a sentence:")
            if sentence =='q' or sentence == 'quit()':
                break
            x_test = process_txt(sentence, word_to_id, args_in_use.seq_length)
            print(x_test)
            feed_dict = {
                test_text: x_test,
                drop: 1.0
            }

            y_pred_cls = sess.run(output, feed_dict=feed_dict)
            print(y_pred_cls)
            y_pred_cls = y_pred_cls[0]
            max_index = np.argmax(y_pred_cls)
            print(id_to_cat[max_index], y_pred_cls[max_index])

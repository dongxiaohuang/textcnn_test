# TEXT CNN test


text cnn model test using pb model

if you have update the data preprocessor method, update `data_preprocessor` method in the `utils/dataprocessor.py`

## batch test
`python text_batch_test.py` to run the batch test
parameters:
  --model: model path
  --dictionary: dictionary path
  --labels: label path, format: labels in the same order of training period, seperated by '\n'
  --seq_length: the length of sequence for text padding
  --testdata: the path of test data
  --tensor_input: the input op_name for graph, format： <op_name>:<output_index>
  --tensor_dropout：the dropout op_name for graph, format： <op_name>:<output_index>
  --tensor_input_keep：the input_keep op_name for graph, format： <op_name>:<output_index>
  --tensor_output：the output op_name for graph, format： <op_name>:<output_index>
  set all the parameters

## case test
`python text_case_test.py` to run the case test
parameters:
  --model: model path
  --dictionary: dictionary path
  --labels: label path, format: labels in the same order of training period, seperated by '\n'
  --seq_length: the length of sequence for text padding
  --tensor_input: the input op_name for graph, format： <op_name>:<output_index>
  --tensor_dropout：the dropout op_name for graph, format： <op_name>:<output_index>
  --tensor_input_keep：the input_keep op_name for graph, format： <op_name>:<output_index>
  --tensor_output：the output op_name for graph, format： <op_name>:<output_index>
  set all the parameters

## To run

- upadte domain testdata in the `data` directory
- update dictionary in the `dict` directory, default tencent pretrained dictionary
- upadte labels in the `lables` directory
- upadte model in the `model` directory
- if you have update the data preprocessor method, update `data_preprocessor` method in the `utils/dataprocessor.py`
- update parameters in `text_case_test` and run case test
- update parameters in `text_batch_test` and run batch test

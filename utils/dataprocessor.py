import tensorflow.contrib.keras as kr
import re
def data_preporcessor(input_str):
    # input_str = ['<Number>' if i.isdigit() else i for i in input_str]
    # alert process
    input_str = ''.join(input_str)
    input_str = re.sub("查询|查一下|查询一下|查下|查询下|查","查询", input_str)
    input_str = re.sub("[0123456789零一二三四五六七八九十]","@", input_str)
    input_str = re.sub("[。，,.：:]","点", input_str)
    input_str = re.sub("删除|取消|关闭|删了|删掉|停止","删除", input_str)
    input_str = re.sub("闹钟|提醒|闹铃|安排|日程","闹钟", input_str)
    input_str = re.sub("凌晨|黎明|清晨|早上|早晨|上午|中午|下午|晚儿|晚上|傍晚|夜里|半夜|深夜|午夜","时段", input_str)
    input_str = re.sub("星期|礼拜","星期", input_str)
    content = list(input_str)
    content = ['<Number>' if a=='@' else a for a in content]
    return content

def process_txt(input_str, word_to_id, seq_length):
    input_str = data_preporcessor(input_str)
    unkIndex = word_to_id.get("<unk>", 1)
    data_id = []
    data_id.append([word_to_id.get(w, unkIndex) for w in input_str])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length, padding='pre')#words for 2d array
    return x_pad

def process_file(source_path, word_to_id, seq_length, cat_to_id):
    """Short summary.
    input file format: text \t label
    Parameters
    ----------
    source_path : type
        Description of parameter `source_path`.
    word_to_id : dictionary
        Description of parameter `word_to_id`.
    seq_length : int
        Description of parameter `seq_length`.
    cat_to_id : dictionary
        Description of parameter `cat_to_id`.

    Returns
    encoded_contents_pad: encoded text(padded to the same length)
    encoded_labels: encoded label
    -------
    type
        Description of returned object.

    """
    encoded_contents = []
    encoded_labels = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip().split('\t')) < 2:
                continue
            content, label = line.strip().split('\t')
            content = data_preporcessor(content)
            unkIndex = word_to_id.get("<unk>", 1)
            encoded_content = [word_to_id.get(word,unkIndex) for word in content]
            encoded_label = cat_to_id[label.strip()]
            encoded_labels.append(encoded_label)
            encoded_contents.append(encoded_content)
        encoded_content_pad = kr.preprocessing.sequence.pad_sequences(encoded_contents, seq_length, padding='pre')
    return encoded_content_pad, encoded_labels

def read_dict(filename):
    word_to_id = {}
    with open(filename, 'r', encoding='utf-8') as f:
        word_to_id = {line.split('\t')[0]: int(line.strip().split('\t')[1]) for line in f}
    return word_to_id

def read_labels(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = [cat.strip() for cat in lines]
        id_to_cat = {id:cat.strip() for id, cat in enumerate(lines)}
        cat_to_id = {cat.strip():id for id, cat in enumerate(lines)}
    return id_to_cat, cat_to_id

def batch_itr(x,y, batch_size):
    data_len = len(x)
    batch_num = int((data_len-1)/batch_size)+1
    for i in range(batch_num):
        start_idx = i*batch_size
        end_idx = min((i+1)*batch_size, data_len)
        yield x[start_idx:end_idx], y[start_idx: end_idx]

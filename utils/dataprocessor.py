import tensorflow.contrib.keras as kr
def process_txt(input_str, word_to_id, seq_length):
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
            content = list(content.strip())
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

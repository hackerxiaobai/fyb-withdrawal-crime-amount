#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    : main.py
@Time    : 2021/11/19 16:02:04
@Author  : Lei Wang
@Contact : lei_wang@shannonai.com
@Version : 0.1
@License : Apache License Version 2.0, January 2004
@Desc    : None
'''
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import  ViterbiDecoder, longest_common_subsequence
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF
import json


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.gpu_options.polling_inactive_delay_msecs = 10
session = tf.compat.v1.Session(config=config)


maxlen = 128
epochs = 20
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 2000  # 必要时扩大CRF层的学习率

# bert配置
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split('\t')
                except:
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D

def json2txt(filename):
    '''
    description:  json 格式化的数据转 txt 数据
    param {*}
    return {*}
    '''    
    dump_data = []
    with open(filename, encoding='utf8') as f:
        data = json.loads(f.read())
        for item in data:
            text = item['text']
            text = list(text)
            labels = ['O']*len(text)
            for e in item['entities']:
                tag = e['tag']
                labels[e['start']] = 'B-' + tag
                for i in range(e['end']-e['start']-1):
                    labels[e['start']+1+i] = 'I-' + tag
            dump_data.append([text, labels])
            
    with open(filename[:-4]+'ner.txt', 'w', encoding='utf8') as f:
        for item in dump_data:
            for t, l in  zip(*item):
                f.write(t+'\t'+l+'\n')
            f.write('\n')
            
                       
json2txt('./data/train.json')
json2txt('./data/valid.json')


# 标注数据
train_data = load_data('./data/train.ner.txt')
valid_data = load_data('./data/valid.ner.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['MONEY']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer.token_to_id(tokenizer._token_start)], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer.token_to_id(tokenizer._token_end)]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='bert'
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output

output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids], [segment_ids]])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1

            json_config = model.to_json()
            with open('model_config.json', 'w') as json_file:
                json_file.write(json_config)

            model.save_weights('model_weights.h5')

        print('valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' % (f1, precision, recall, self.best_val_f1))
        
def main():
    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

if __name__ == '__main__':

    main()


#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    : utils.py
@Time    : 2021/11/19 16:02:09
@Author  : Lei Wang
@Contact : lei_wang@shannonai.com
@Version : 0.1
@License : Apache License Version 2.0, January 2004
@Desc    : None
'''

import re
import sys
import json
import random

import cn2an

from copy import deepcopy
from tqdm import tqdm

# 初始化
maxlen = 1000


class ZeroOneBackPack():
    """[零一完全背包，找出哪些金额子句和才是正好等于真实金额数]
    """
    def __init__(self, items, money) -> None:
        self.items = items
        self.ret = []
        self.money = money

    def master(self):
        if self.work(self.money, 0):
            return self.ret

    def work(self, weight, i):
        for j in range(i, len(self.items)):
            if self.items[j][0] < weight:
                if self.work(float('%.3f' % (weight - self.items[j][0])),
                             j + 1):
                    self.ret.append(self.items[j])
                    return True
            elif self.items[j][0] == weight:
                self.ret.append(self.items[j])
                return True
        return False


def get_all_money(str):
    """[正则抽取金额]

    Args:
        str ([str]): [包含金额的句子]

    Returns:
        [object]: [抽取到的所有金额]
    """
    rule = '([0-9]{1,10}(\.[0-9]{0,4})?(多|余)?(千|万)?(多|余)?元)'
    match = re.findall(rule, str)
    if match:
        return match
    else:
        return ''


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """[将文本按照标点符号划分为若干个短句]

    Args:
        text ([str]): [输入长句]
        maxlen ([int]]): [保留最长长度]
        seps (str, optional): [分隔符]. Defaults to '\n'.
        strips ([str], optional): [句子前后去除字符]. Defaults to None.

    Returns:
        [list]: [分割后的一些子句]
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def text_split(text, limited=True):
    """[将长句按照标点分割为多个子句。]

    Args:
        text ([str]): [输入长句]
        limited (bool, optional): [子句长度限制]. Defaults to True.
 
    Returns:
        [list]: [分割后的一些子句]
    """
    texts = text_segmentate(text, 1, u'\n。；：，;、')
    if limited:
        texts = texts[-maxlen:]
    return texts


def load_data(filename, train=True):
    """[加载数据]

    Args:
        filename ([str]): [数据路径]
        train (bool, optional): [区分训练集还是测试集]. Defaults to True.

    Returns:
        [list]: [
                    数据集返回，list里放的是元组，
                    [
                        (sentence, label),
                        (sentence, label),
                        ...
                    ]
                ]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            l['justice'] = l['justice'].replace('X', '').replace(
                '&ldquo', '').replace('&rdquo', '')
            if train:
                D.append([l['justice'], l['money']])
            else:
                D.append([l['justice'], 100])

    # 处理类似 三十六元 这种case 转成 36元
    rule = '([数,零,一,二,三,四,五,六,七,八,九,十,百,千]{1,10}(多|余)?(万)?(多|余)?元)'
    for item in D:
        all_match = re.findall(rule, item[0])
        if all_match:
            for the_money in all_match:
                if the_money[0] in ['千元', '万元', '数百元', '一二百元', '百元']:
                    continue
                change_money = the_money[0].replace("多", "").replace("余", "")
                try:
                    num_money = cn2an.cn2an(change_money, "smart")
                    item[0] = item[0].replace(the_money[0], str(num_money) + '元')
                except:
                    pass

    # 处理 3千元 3.5万元等
    rule = '([0-9]{1,10}(\.[0-9]{0,4})?(多|余)?(千|万)?(多|余)?元)'
    for item in D:
        all_match = re.findall(rule, item[0])
        if all_match:
            for the_money in all_match:
                change_money = the_money[0].replace("多", "").replace("余", "")
                if "万元" in change_money:
                    num_money = str(
                        float(change_money.replace('万元', '')) * 10000)
                    if num_money[-1] == '0' and num_money[-2] == '.':
                        num_money = num_money[:-2]
                    item[0] = item[0].replace(the_money[0], num_money + '元')

                elif "千元" in change_money:
                    num_money = str(
                        float(change_money.replace('千元', '')) * 1000)
                    if num_money[-1] == '0' and num_money[-2] == '.':
                        num_money = num_money[:-2]
                    item[0] = item[0].replace(the_money[0], num_money + '元')
                else:
                    item[0] = item[0].replace(the_money[0], change_money)
    return D


def extract_flow(inputs):
    """[正则抽取金额]

    Args:
        inputs ([tuple]): [输入长句，总金额数]

    Returns:
        [tuple]: [所有包含金额的子句，总金额数]
    """
    text, label = inputs
    texts = text_split(text, False)

    # 包含所有有数字的句子
    pattern = re.compile('[0-9]+')
    texts = [i for i in texts if pattern.findall(i)]

    # 金额相关的所有句子
    rule = '([0-9]{1,10}(\.[0-9]{0,4})?(多|余)?(万)?(多|余)?元)'
    textss = [i for i in texts if re.findall(rule, i)]

    return textss, label


def get_subsent_money_index(data):
    """[找到金额对应的子句索引]

    Args:
        data ([list]):[所有子句]

    Returns:
        [list]: [金额对应索引]
    """
    results = []
    for index, item in enumerate(data):
        tmp_multi_money_count = []
        tmp_money_count = 0.00
        label = item[1]
        for sub_index, text in enumerate(item[0]):
            all_money = get_all_money(text)
            for the_money in all_money:
                change_money = the_money[0].replace("多", "").replace("余", "")
                if "万元" in change_money:
                    tmp_money_count = float(change_money.replace('万元','')) * 10000
                elif "千元" in change_money:
                    tmp_money_count = float(change_money.replace('千元','')) * 1000
                else:
                    tmp_money_count = float(change_money.replace('元', ''))
                tmp_multi_money_count.append([tmp_money_count, sub_index])
                tmp_money_count = 0.00
        results.append((tmp_multi_money_count, label, index))
    return results


def build_ner_data(match_right, data):
    all_data = []

    for ma, source in zip(match_right, data):
        tmp_data = {"text": "".join(source[0]), "entities": []}

        if not ma[0]:
            all_data.append(tmp_data)
            continue

        for index, sub_sent in enumerate(source[0]):
            if index == 0:
                iter_sent_length = 0
            else:
                iter_sent_length += len(source[0][index - 1])

            iterlist = []
            for m in ma[0]:
                if index == m[1]:
                    if m[0] in iterlist:
                        continue
                    else:
                        iterlist.append(m[0])

                    m_money = str(m[0])
                    # 整数还原到原句
                    if m_money[-1] == '0' and m_money[-2] == '.':
                        m_money = m_money[:-2] + '元'
                    # 小数还原到原句
                    if m_money[-1] != '0' and m_money[-2] == '.':
                        if sub_sent[sub_sent.index(m_money) +
                                    len(m_money)] == '0':
                            m_money = m_money + '0元'

                    # 这一步会将所有相同金额数遍历完的，所以需要上面的if else
                    for p in re.finditer(m_money, sub_sent):
                        if p.start() != 0 and sub_sent[p.start() - 1] in [
                                '0', '1', '2', '3', '4', '5', '6', '7', '8',
                                '9'
                        ]:
                            continue
                        tmp_data['entities'].append(
                                {
                                    "term": m_money,
                                    "tag": "MONEY",
                                    "start": p.start() + iter_sent_length,
                                    "end": p.end() + iter_sent_length
                                }
                            )

        all_data.append(tmp_data)
    return all_data


def main():
    """[通过零一完全背包去找到真确的金额，并构造出NER标注数据]

    Args:
        data ([str]): [训练集路径]
    """

    raw_data = load_data(input_path)

    filter_money_data = []
    for item in raw_data:
        filter_money_data.append(extract_flow(item))

    all_results = get_subsent_money_index(filter_money_data)

    # 找出零一完全背包恰好等于总金额的数据
    match_right = []
    for index, item in enumerate(all_results):
        ret = ZeroOneBackPack(item[0], eval(item[1])).master()
        if ret:
            ret = sorted(ret, key=lambda x: x[-1])
        match_right.append([ret, item[-1]])

    # 构造ner数据
    all_data = build_ner_data(match_right, filter_money_data)

    # 划分训练集测试集
    random.shuffle(all_data)
    train = []
    valid = []

    for index in range(len(all_data)):
        if index % 13 == 0:
            valid.append(all_data[index])
        else:
            train.append(all_data[index])

    with open(save_path + 'train.json', "w", encoding="utf-8") as fw:
        fw.write(
            json.dumps(train,
                       ensure_ascii=False,
                       indent=4,
                       separators=(',', ':')))

    with open(save_path + 'valid.json', "w", encoding="utf-8") as fw:
        fw.write(
            json.dumps(valid,
                       ensure_ascii=False,
                       indent=4,
                       separators=(',', ':')))


if __name__ == "__main__":
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    main()
is_debug = False


def dbg_print(x):
    if is_debug:
        print(x)


# init the vocabulary
vocabulary = [chr(x) for x in range(256)]
vocabulary.append("<|endoftext|>")
dbg_print(len(vocabulary))
dbg_print(vocabulary)

# train data
text = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# import regex as re
# pretoken = re.findall(PAT, text)

pretoken = text.split()
dbg_print(pretoken)

data_table = {}
pretoken_set = set(pretoken)
for item in pretoken_set:
    data_table[tuple(list(item))] = pretoken.count(item)

dbg_print(data_table)

delete_key = []
for key, value in data_table.items():
    if len(key) <= 1:
        delete_key.append(key)

for key in delete_key:
    data_table.pop(key)

print(data_table)


def sliding_pair_iterator(tuple_data):
    """用迭代器实现滑动取连续两个元素"""
    # 将元组转为迭代器
    it = iter(tuple_data)
    try:
        # 先取出第一个元素作为上一个元素（锚点）
        prev = next(it)
    except StopIteration:
        # 空元组直接返回
        return

    # 遍历剩余元素
    for current in it:
        # 返回当前配对（上一个+当前）
        yield (prev, current)
        # 更新锚点为当前元素，供下一次配对
        prev = current


i = 6
while i > 0:
    tmp_dict = {}
    for key, value in data_table.items():
        for pair in sliding_pair_iterator(key):
            # print(pair)
            pair_key = pair[0] + pair[1]
            tmp_dict[pair_key] = tmp_dict.get(pair_key, 0) + value
    # print(tmp_dict)

    i -= 1

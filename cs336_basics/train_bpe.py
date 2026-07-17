import os
import re
import regex
from pretokenization_example import find_chunk_boundaries
from typing import List
from pathlib import Path

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
) -> tuple[dict[int, bytes], List[tuple[bytes, bytes]]] :
    """
    在给定数据集上训练BPE分词器：
    1.将数据集中的文本读取为字符串，匹配特殊token，分割文本
    2.初始化vocab字典，将0～255的ASCII字符加入vocab字典
    3.使用GPT-2的正则表达式对文本进行分词
    4.在预分词的每个单词内，统计相邻字节的出现次数，存储在pair_count字典中
    5.merge:选择出现次数最多的相邻字节对，组成新token

    
    """
    if not Path(input_path).exists() :
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
    
    with open(input_path, "rb") as f :
        text = f.read()
    
## 建立pattern， 用逻辑或‘｜’来连接所有的特殊token， 
## 并且使用re.escape()来转义特殊字符， 以确保它们在正则表达式中被正确处理。 
## 另外，使用sorted()函数按长度降序排列特殊token，以防止某些特殊token是另一些的子串
## train_segments是一个列表，包含了所有分割后的文本段落，去掉了空字符串
    if special_tokens :
        pattern = '|'.join(
            re.escape(stoken) 
            for stoken in sorted(special_tokens, key = len, reverse = True)
        )
        train_segments = [segment for segment in re.split(pattern, text.decode("utf-8", errors="ignore")) if segment]
    else :
        train_segments = [text.decode("utf-8", errors="ignore")]

## 初始化vocab字典，将0～255的ASCII字符加入vocab字典
## bytes(List[int])将整数列表转换为字节串
    vocab = {i: bytes([i]) for i in range(256)}
## 将特殊token加入列表
    ##for stoken in special_tokens :
    ##    if stoken not in vocab.values() :
    ##        vocab[len(vocab)] = stoken.encode("utf-8")


## 使用re.compile()来编译正则表达式模式，如果不编译的话，后面每次都要把正则表达式传入函数
## 编译后适合多次使用
    pre_tokenize_pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    raw_count = {}
    '''
        segments是一个列表,包含了所有分割后的文本段落,去掉了空字符串,例如["Hello,world"]
        tokens是一个列表,用GPT-2的正则表达式对文本进行分词,得到字符串列表,例如["Hello", ",", "world"]
        token_bytes是一个元组,包含了每个分词的字节表示,例如[('H',), ('e',), ('l',), ('l',), ('o',)]
        raw_count是一个字典,键是token_bytes,值是该token_bytes在所有segments中出现的次数
        为什么非要把token_bytes变成元组呢?因为字典的键必须是可哈希的,而列表是不可哈希的,所以要把列表转换为元组
    '''
    for segment in train_segments :
        tokens = pre_tokenize_pattern.findall(segment)
        for token in tokens :
            token_bytes = tuple(bytes([c]) for c in token.encode("utf-8"))
            raw_count[token_bytes] = raw_count.get(token_bytes, 0) + 1
    

## 我们已经把整个训练文本分词并统计了出现次数，存在raw_count字典中，key是tuple
## 现在要统计相邻词的出现次数，以将它们合并为一个新词。我们使用一个字典pair_counts来存储相邻词对的出现次数
## pair_count的键是一个元组，包含两个相邻词的字节表示，值是该相邻词对在所有tokens中出现的次数
    pair_count = {} 
    word_list = [] # 将raw_count的键转换为列表
    count_list = [] # 将raw_count的值转换为列表
    indices = {} # 用于存储每个pair在word_list中的索引
    idx = 0
    for token_bytes, count in raw_count.items() :
        word = list(token_bytes)
        word_list.append(word)
        count_list.append(count)
        for i in range(len(word) - 1) :
            pair = (word[i], word[i + 1])
            pair_count[pair] = pair_count.get(pair, 0) + count
            if pair not in indices:
                indices[pair] = set()
            indices[pair].add(idx)
        idx += 1

## 接下来进行merge操作
## 刚刚的预分词是单词级别的，用于限制BPE合并边界，避免跨单词或标点进行merge。
## merge采用字节级别，可覆盖任意UTF-8文本，并从基础字节逐步学习常见子词.
    merge_count = vocab_size - len(vocab) - len(special_tokens)
    merges : List[tuple[bytes, bytes]] = []
    for i in range(merge_count) :
        best_pair = max(pair_count, key = lambda pair : (pair_count[pair], pair))
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)
    ## 接下来需要更新pair_count才可以继续merge
        if not pair_count :
            break
    ## 遍历每一个出现了best_pair的单词,indices后面会改变，所以复制一下改成list局部使用
        indices_copy = list(indices[best_pair])
        for idx in indices_copy :
            word = word_list[idx]
            count = count_list[idx]
            i = 0
            ## 去单词里找到出现best_pair的地方
            while i < len(word) - 1 :
                ## 找到了就开始更新pair_count,首先要把new_token对应的那个pair的次数减去count次
                ## 如果减完之后这个pair没有次数了，就要在pair_count里面删掉这个key
                ## 还要改变左邻居和右邻居的派人_count
                ## 例如hello里面(e，l)是best_pair的话，就要把(e,l)组合的次数减掉，把(h,el)和(el,l)的次数加上
                ## 加入新的组合之后，还要更新word和indices，例如本来是[H,e,l,l,o],现在是[H,el,l,o]
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1] :
                    ## 左邻居
                    if i > 0 :
                        left_pair = (word[i - 1], word[i])
                        pair_count[left_pair] -=count
                        if pair_count[left_pair] <= 0 :
                            del pair_count[left_pair]

                        ## 新token和左边一个token组成了新pair
                        new_pair = (word[i - 1], new_token)
                        pair_count[new_pair] = pair_count.get(new_pair, 0) + count
                        if new_pair not in indices:
                            indices[new_pair] = set()

                        ## 更新indices
                        indices[new_pair].add(idx)
                    
                    ## 右邻居，各操作同上
                    if i < len(word) - 2 :
                        right_pair = (word[i], word[i + 1])
                        pair_count[right_pair] -=count
                        if pair_count[right_pair] <= 0 :
                            del pair_count[right_pair]

                        new_pair = (new_token, word[i + 2])
                        pair_count[new_pair] = pair_count.get(new_pair, 0) + count
                        if new_pair not in indices:
                            indices[new_pair] = set()

                        indices[new_pair].add(idx)
                    
                    ## 两个token合并了
                    word[i] = new_token
                    word.pop(i + 1)
                    i +=1
                ## 没找到就继续右移
                else :
                    i +=1
        ## 遍历了所有出现过best_pair的词之后，best_pair要被删除掉
        if best_pair in indices :
            del indices[best_pair]
        if best_pair in pair_count :
            del pair_count[best_pair]
    ## merge的轮数已经达到要求了，merge过程中已经更新了vocab，现在把特殊字符加进去
    for stoken in special_tokens :
        if stoken.encode("utf-8") not in vocab.values() :
            vocab[len(vocab)] = stoken.encode("utf-8")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    return (vocab, merges)
## 此时就会发现，新的pair merge出来之后，例如把‘a’‘b’合并了
## 如果只是把pair_count中的（a，b）这一对删掉，新的token‘ab’和其相邻的字节的pair无法被统计
## 只能重新遍历文本，把所有的pair重新统计一遍，才能得到新的pair_count，而且这个遍历不能是字节级别的
## indices就是为了记录每个pair在word_list中的索引，方便我们在合并后更新pair_count而不必重新遍历



        
## 下方代码块使用使用作业提供的pretokenization_example.py来把文件分成指定数量的chunk
## 这是由于pretokenization可能成为性能瓶颈，为了方便并行计算，在每个chunk上独立地进行BPE训练，最后再合并结果
## 该函数会按照要求的块数对文件进行分割，并且会自动的识别特殊token
## 尽量让每个块的边界都在特殊token的边界上，避免在文本中间进行分割，造成文本意义的不连贯

'''
input_path  = current_dir.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"                    
with open(input_path, "rb") as f :
        text = f.read()
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print(boundaries)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            print(chunk)
'''
## 为了程序的可迁移性，把PROJECT_ROOT定义为当前文件的父目录的父目录
## 这样无论在什么环境下运行，都会从项目根目录开始寻找数据文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent
## input_path = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
input_path = PROJECT_ROOT / "data" / "test.txt"


train_bpe(
    input_path = input_path,
    vocab_size = 1000,
    special_tokens = ["<|endoftext|>", "<|pad|>"]
)
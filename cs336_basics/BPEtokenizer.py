import re
import regex
import json
from pathlib import Path
from typing import Iterable, Iterator

class BPEtokenizer :
    """
    定义tokenize类，实现encode和decode功能
    """
    def __init__(self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = {pair: id for id, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        self.byte_to_id = {b: i for i, b in vocab.items()}
        # 规定正则模式,先将special_tokens用长度从长到短排序
        # 用｜（or）连接special_tokens，用于从长文本中提取出来一句一句话
        if special_tokens :
            self.special_pattern = '|'.join(
                re.escape(stoken) 
                for stoken in sorted(special_tokens, key = len, reverse = True)
            )
            self.special_pattern = "(" + self.special_pattern + ")"
            self.special_pattern = re.compile(self.special_pattern)
        else :
            self.special_pattern = None
        # 规定gpt-2正则，用于预分词，把一句话给分割成单词和标点符号
        self.pre_tokenize_pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
        
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {int(k):v.encode("latin-1") for k,v in vocab.items()}

        with open(merges_filepath, encoding = "utf-8") as f:
            merges = json.load(f)
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in merges]

        return cls(vocab, merges, special_tokens)
    

    # 定义encode方法，输入text字符串，提供merges和vocab，将text转化为token id的列表
    # 1. 先按照特殊token的正则捕获所有特殊字符，用split，得到的是一个个短字符串(segments)，被特殊字符分隔
    # 2. 这里必须带括号(捕获组)得到的字符串列表里面包含了特殊字符
    # 3. 然后遍历字符串列表里面的segment，每个segment分为两类
    # 如果是特殊token，就直接append进token化的列表里；
    # 如果不是，就取出来，按照gpt-2规则分离单词和标点，对于分离出的每个碎片再根据merge规则转换为token id

    def encode(self, text: str) -> list[int]:
        segment_id = []
        if not text:
            return []
        if not self.special_tokens:
            tokens = self.pre_tokenize_pattern.findall(text)
            for token in tokens:
                idx = self.word_merge(token)
                for id in idx:
                    segment_id.append(id)
            return segment_id
        segments = [segment for segment in re.split(self.special_pattern, text) if segment]
        for segment in segments:
            if segment not in self.special_tokens:
                tokens = self.pre_tokenize_pattern.findall(segment)
                for token in tokens:
                    idx = self.word_merge(token)
                    for id in idx:
                        segment_id.append(id)
            else:
                segment_id.append(self.byte_to_id[bytes(segment.encode("latin-1"))])
        return segment_id


    # 执行在单词中查找merge pair然后合并的操作
    def word_merge(self, word: str) -> list[int]:
        idx = []
        bytes_word = [bytes([b]) for b in word.encode("utf-8")]
        while len(bytes_word) >= 2:
            best_pair = None 
            best_rank = float('inf')
            for i in range(len(bytes_word) - 1):
                pair = (bytes_word[i], bytes_word[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair == None:
                break
            new_bytes_word = []
            i = 0
            while i < len(bytes_word):
                if i < len(bytes_word) - 1 and best_pair == (bytes_word[i], bytes_word[i + 1]):
                    new_bytes_word.append(best_pair[0] + best_pair[1])
                    i +=2
                else:
                    new_bytes_word.append(bytes_word[i])
                    i +=1
            bytes_word = new_bytes_word
        for byte in bytes_word:
            idx.append(self.byte_to_id[byte])
        return idx

    # 迭代编码器
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    # 解码器
    def decode(self, encoded_text: list[int]) -> str:
        if not encoded_text:
            return ""
        byte_segments = [self.vocab[i] for i in encoded_text]
        decoded_text = b"".join(byte_segments)
        return decoded_text.decode("utf-8", errors = "replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_path = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
data_path = PROJECT_ROOT / "data" 
tokenizer = BPEtokenizer.from_files(
    data_path / "vocab.json",
    data_path / "merges.json",
    special_tokens = ["<|endoftext|>"]
    )
encoded_text = []
encoded_ids = tokenizer.encode("s")
decoded_string = tokenizer.decode(encoded_ids)
with open(input_path, "r", encoding = "utf-8") as f:
    generator = tokenizer.encode_iterable(f)
    while True:
        try:
            segment_id = next(generator)
            encoded_text.append(segment_id)
        except StopIteration:
            break



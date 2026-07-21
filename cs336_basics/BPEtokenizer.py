from train_bpe import train_bpe
import re
import regex


class BPEtokenize :
    """
    定义tokenize类，实现encode和decode功能
    """
    def __init__(self, 
        vocab: dict[bytes, int], 
        merges: List[tuple[bytes, bytes]], 
        special_tokens: List[str] | None = None
    ):
        self.vocab = vocab
        self.merges = {pair: id for id, pair in enumerate(merges)}
        self.special_tokens = special_tokens
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

    def from_files(
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None
    ):
        
    

    # 定义encode方法，输入text字符串，提供merges和vocab，将text转化为token id的列表
    # 1. 先按照特殊token的正则捕获所有特殊字符，用split，得到的是一个个短字符串(segments)，被特殊字符分隔
    # 2. 这里必须带括号(捕获组)得到的字符串列表里面包含了特殊字符
    # 3. 然后遍历字符串列表里面的segment，每个segment分为两类
    # 如果是特殊token，就直接append进token化的列表里；
    # 如果不是，就取出来，按照gpt-2规则分离单词和标点，对于分离出的每个碎片再根据merge规则转换为token id

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if not self.special_tokens:
            return 
        segments = [segment for segment in re.split(self.special_pattern, text) if segment]
        text_seperate_word = []
        for segment in segments:
            if segment not in self.special_tokens:
                tokens = self.pre_tokenize_pattern.findall(segment)
                for token in tokens:
                    
            else:

                text_seperate_word.append(segment)

    # 执行在单词中查找merge pair然后合并的操作
    def word_merge(self, word: str) -> List[int]:
        idx = []
        bytes_word = [bytes([b]) for b in word.encode("utf-8")]
        while len(bytes_word) >= 2:
            best_pair = None 
            best_rank = float('inf')
            for i in range(len(bytes_word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair == None:
                break
            new_bytes_word = []
            while i < len(bytes_word):
                if i < len(bytes_word - 1) and best_pair == (bytes_word[i], bytes_word[i + 1]):
                    new_bytes_word.append(best_pair[0] + best_pair[1])
                    i +=2
                else:
                    new_bytes_word.append[bytes_word[i]]
                    i +=1
            bytes_word = new_bytes_word
        for bytes in bytes_word:
            idx.append(self.merges[bytes])
        return idx


        

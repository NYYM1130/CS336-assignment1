from train_bpe import train_bpe
import re
import regex


class BPEtokenize :
    """
    定义tokenize类，实现encode和decode功能
    """
    def __init__(self, 
        vocab: dict[bytes, int], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
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

#    def from_files(
#        vocab_filepath: str,
#        merges_filepath: str,
#        special_tokens: list[str] | None = None
#   ):
        
    

    # 定义encode方法，输入text字符串，提供merges和vocab，将text转化为token id的列表
    # 1. 先按照特殊token的正则捕获所有特殊字符，用split，得到的是一个个短字符串(segments)，被特殊字符分隔
    # 2. 这里必须带括号(捕获组)得到的字符串列表里面包含了特殊字符
    # 3. 然后遍历字符串列表里面的segment，每个segment分为两类
    # 如果是特殊token，就直接append进token化的列表里；
    # 如果不是，就取出来，按照gpt-2规则分离单词和标点，对于分离出的每个碎片再根据merge规则转换为token id

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.special_tokens:
            return 
        segments = [segment for segment in re.split(self.special_pattern, text) if segment]
        word_id = []
        for segment in segments:
            if segment not in self.special_tokens:
                tokens = self.pre_tokenize_pattern.findall(segment)
                for token in tokens:
                    idx = self.word_merge(token)
                    for id in idx:
                        word_id.append(id)
            else:
                word_id.append(self.vocab[bytes(segment)])


    # 执行在单词中查找merge pair然后合并的操作
    def word_merge(self, word: str) -> list[int]:
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

BPEtokenize.encode("Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!  
He said, “Wow, that is a really amazing vase! Can I buy it?” 
The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. 
And that's how Ben found an amazing vase in the store!
<|endoftext|>
Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
<|endoftext|>
One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.
Then, something unexpected happened. The tiger started to shake. Tim was scared. He did not know what was going on. But then, the tiger turned into a nice dog. Tim was very surprised.
Tim and the dog played together now. They were very happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.
<|endoftext|>

Once upon a")
        

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import html
import string

import ftfy
import regex as re
from transformers import AutoTokenizer

__all__ = ['HuggingfaceTokenizer']


def basic_clean(text):
    """
    对输入文本进行基本的清理，包括修复文本中的常见问题和解码 HTML 实体。

    参数:
        text (str): 输入的文本字符串。

    返回:
        str: 清理后的文本字符串。
    """
    # 使用 ftfy.fix_text 修复常见的文本编码问题，如乱码或错误的字符编码
    text = ftfy.fix_text(text)
    # 使用 html.unescape 解码 HTML 实体两次，以确保所有 HTML 实体都被正确解码
    text = html.unescape(html.unescape(text))
    # 去除文本首尾的空白字符
    return text.strip()


def whitespace_clean(text):
    """
    对输入文本进行空白字符的清理，将多个连续的空白字符替换为单个空格，并去除首尾空白。

    参数:
        text (str): 输入的文本字符串。

    返回:
        str: 清理后的文本字符串。
    """
    # 使用正则表达式将多个连续的空白字符（包括空格、制表符、换行符等）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除文本首尾的空白字符
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    """
    对输入文本进行规范化和清理，包括替换下划线、去除标点符号、小写化以及压缩空白字符。

    参数:
        text (str): 输入的文本字符串。
        keep_punctuation_exact_string (Optional[str]): 可选的字符串，用于指定在去除标点符号时保留的分隔符。如果提供，则仅在该分隔符处保留标点符号。

    返回:
        str: 规范化和清理后的文本字符串。
    """
    # 将文本中的下划线替换为空格
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        # 如果提供了分隔符，则在分隔符处分割文本，去除每个部分中的标点符号，然后重新拼接
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        # 否则，直接去除文本中的所有标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
    # 将文本转换为小写
    text = text.lower()
    # 使用正则表达式将多个连续的空白字符替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除文本首尾的空白字符
    return text.strip()


class HuggingfaceTokenizer:
    """
    HuggingfaceTokenizer 类封装了 Hugging Face 的 AutoTokenizer，用于文本的分词和编码。
    """
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        """
        初始化 HuggingfaceTokenizer。

        参数:
            name (str): 分词器的名称或路径，例如 'bert-base-uncased'。
            seq_len (Optional[int], 可选): 序列的最大长度。如果为 None，则不进行截断或填充。
            clean (Optional[str], 可选): 文本清理的类型，可选 'whitespace', 'lower', 'canonicalize'。如果为 None，则不进行清理。
            **kwargs: 其他关键字参数，用于传递给 AutoTokenizer。
        """
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # 初始化分词器
        # 从预训练模型加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        # 获取词汇表大小
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        """
        对输入序列进行分词和编码。

        参数:
            sequence (str 或 List[str]): 输入的文本字符串或字符串列表。
            **kwargs: 其他关键字参数，用于传递给分词器的编码方法。

        返回:
            torch.Tensor: 编码后的 token ID 张量。
        """
        # 检查是否需要返回注意力掩码
        return_mask = kwargs.pop('return_mask', False)

        # 设置分词器的参数
        _kwargs = {'return_tensors': 'pt'}  # 设置返回的张量类型为 PyTorch 张量
        if self.seq_len is not None:
            # 如果指定了序列长度，则启用填充和截断，并设置最大长度
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)  # 更新其他参数

        # tokenization
        if isinstance(sequence, str):
            # 如果输入是单个字符串，则转换为列表
            sequence = [sequence]
        if self.clean:
            # 对每个字符串进行清理
            sequence = [self._clean(u) for u in sequence]

        # 对序列进行分词和编码
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            # 如果需要返回注意力掩码，则返回输入 ID 和注意力掩码
            return ids.input_ids, ids.attention_mask
        else:
            # 否则，只返回输入 ID
            return ids.input_ids

    def _clean(self, text):
        """
        对输入文本进行清理。

        参数:
            text (str): 输入的文本字符串。

        返回:
            str: 清理后的文本字符串。
        """
        if self.clean == 'whitespace':
            # 如果清理类型为 'whitespace'，则先进行基本清理，然后替换多个空白字符为单个空格，并去除首尾空白
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            # 如果清理类型为 'lower'，则先进行基本清理和空白清理，然后转换为小写
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            # 如果清理类型为 'canonicalize'，则进行规范化和清理
            text = canonicalize(basic_clean(text))
            
        return text

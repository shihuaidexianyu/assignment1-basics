import json
from collections.abc import Iterable
from functools import lru_cache


try:
    import regex as re

    _GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
except ImportError:  # pragma: no cover - fallback for environments without regex
    import re

    print("Warning: 'regex' module not found, using standard 're'. Pre-tokenization might differ.")
    _GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


class tokenizer:
    """A byte-level BPE tokenizer compatible with GPT-2 vocab/merges."""

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self._bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self._merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self._sorted_special_tokens = sorted(set(self.special_tokens), key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        # 先实例化一个对象,然后调用该方法初始化
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = {token_id: token_bytes.encode("utf-8") for token_bytes, token_id in vocab_data.items()}
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                token_a_str, token_b_str = line.strip().split()
                merges.append((token_a_str.encode("utf-8"), token_b_str.encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> list[str]:
        """Split text into pre-tokens using the GPT-2 regex."""
        # 之前的错误：这里直接把 regex 结果 encode 成 bytes，
        # 导致后续 BPE 把“整段 bytes”当成一个 token 来合并，unicode 会失配。
        return list(_GPT2_PATTERN.findall(text))

    def _iter_special_segments(self, text: str) -> list[tuple[bool, str]]:
        # 之前的错误：没有先切分 special token，导致 <|endoftext|> 被 regex 切碎，
        # 最终找不到 vocab 里的整 token。
        if not self._sorted_special_tokens:
            return [(False, text)]

        segments: list[tuple[bool, str]] = []
        idx = 0
        start = 0
        while idx < len(text):
            matched = False
            for token in self._sorted_special_tokens:
                if text.startswith(token, idx):
                    if start < idx:
                        segments.append((False, text[start:idx]))
                    segments.append((True, token))
                    idx += len(token)
                    start = idx
                    matched = True
                    break
            if not matched:
                idx += 1
        if start < len(text):
            segments.append((False, text[start:]))
        return segments

    @staticmethod
    def _get_pairs(symbols: list[bytes]) -> set[tuple[bytes, bytes]]:
        return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}

    @lru_cache(maxsize=10000)
    def _bpe(self, token_bytes: bytes) -> tuple[bytes, ...]:
        # 之前的错误：按 merges 列表“全局反复扫一遍”合并，
        # 而不是按 merge rank 在当前 token 内部逐步合并。
        # 正确做法是：始终选 rank 最小的 pair 进行合并。
        if not token_bytes:
            return tuple()
        if len(token_bytes) == 1:
            return (token_bytes,)

        symbols = [bytes([b]) for b in token_bytes]
        pairs = self._get_pairs(symbols)
        while pairs:
            best_pair = None
            best_rank = None
            for pair in pairs:
                rank = self._merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            new_symbols: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break
            pairs = self._get_pairs(symbols)
        return tuple(symbols)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for is_special, segment in self._iter_special_segments(text):
            if is_special:
                special_bytes = segment.encode("utf-8")
                token_id = self._bytes_to_id.get(special_bytes)
                if token_id is None:
                    raise ValueError(f"Special token {segment} not found in vocabulary.")
                token_ids.append(token_id)
                continue

            for token in self.pretokenize(segment):
                token_bytes = token.encode("utf-8")
                for piece in self._bpe(token_bytes):
                    token_id = self._bytes_to_id.get(piece)
                    if token_id is None:
                        raise ValueError(f"Token {piece} not found in vocabulary.")
                    token_ids.append(token_id)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # 之前的错误：yield 出 list[int]，但测试期望逐个 token id 流式输出。
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        byte_chunks = [self.vocab[token_id] for token_id in token_ids]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

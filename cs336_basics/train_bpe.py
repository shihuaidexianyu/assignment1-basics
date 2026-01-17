import os
import multiprocessing
import collections
from typing import BinaryIO, List, Tuple, Dict
import functools

try:
    from cs336_bpe_rs import pretokenize_and_count as rust_pretokenize_and_count
    from cs336_bpe_rs import train_bpe as rust_train_bpe
    from cs336_bpe_rs import pretokenize_and_count_from_buffer as rust_pretokenize_and_count_from_buffer
    from cs336_bpe_rs import train_bpe_from_buffer as rust_train_bpe_from_buffer

    print("use rust extension for bpe training")
except Exception:
    rust_pretokenize_and_count = None
    rust_train_bpe = None
    rust_pretokenize_and_count_from_buffer = None
    rust_train_bpe_from_buffer = None

# 尝试导入 regex 库 (支持 \p{L} 等高级特性)，如果不存在则回退到 re
try:
    import regex as re

    # GPT-2 pattern provided in assignment
    GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
except ImportError:
    import re

    print("Warning: 'regex' module not found, using standard 're'. Pre-tokenization might differ.")
    GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    filename: str,
    desired_num_chunks: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> List[int]:
    """
    计算文件分块边界，尽量对齐到特殊 token，避免切断 token。
    若找不到特殊 token，则退化为单 chunk。
    """
    file_size = os.path.getsize(filename)
    if file_size == 0:
        return [0]
    if not split_special_token:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    with open(filename, "rb") as f:
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)
            while True:
                mini_chunk = f.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _process_chunk_worker(args) -> collections.Counter:
    """
    Worker function: 读取文件的一个片段，正则分词，返回词频统计。
    """
    filename, start, end, special_tokens = args
    local_counts = collections.Counter()

    # 也就是作业中提到的 "Pre-tokenization" 步骤
    try:
        with open(filename, "rb") as f:
            f.seek(start)
            # 读取字节并 decode
            # errors='ignore' 防止边界处切断了多字节字符导致报错
            text = f.read(end - start).decode("utf-8", errors="ignore")

        if rust_pretokenize_and_count is not None:
            for token_bytes, count in rust_pretokenize_and_count(text, special_tokens):
                local_counts[tuple(token_bytes)] += count
        else:
            for segment in _iter_non_special_segments(text, special_tokens):
                # 1. 正则切分
                tokens = re.findall(GPT2_PAT, segment)

                # 2. 转换为字节元组并统计
                # 例如: "Hello" -> b"Hello" -> (72, 101, 108, 108, 111)
                for token in tokens:
                    token_bytes = tuple(token.encode("utf-8"))
                    local_counts[token_bytes] += 1

    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {e}")

    return local_counts


def _iter_non_special_segments(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]

    segments: List[str] = []
    tokens = sorted(set(special_tokens), key=len, reverse=True)
    idx = 0
    start = 0
    while idx < len(text):
        matched = False
        for token in tokens:
            if text.startswith(token, idx):
                if start < idx:
                    segments.append(text[start:idx])
                idx += len(token)
                start = idx
                matched = True
                break
        if not matched:
            idx += 1
    if start < len(text):
        segments.append(text[start:])
    return segments


def get_stats(vocab_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
    统计当前所有词汇中的 pair 频率
    """
    pairs = collections.defaultdict(int)
    for ids, freq in vocab_counts.items():
        for i in range(len(ids) - 1):
            pairs[ids[i], ids[i + 1]] += freq
    return pairs


def merge_vocab(
    best_pair: Tuple[int, int], vocab_counts: Dict[Tuple[int, ...], int], new_token_id: int
) -> Dict[Tuple[int, ...], int]:
    """
    将词表中所有的 best_pair 替换为 new_token_id
    """
    new_vocab = {}
    p0, p1 = best_pair

    for ids, freq in vocab_counts.items():
        # 优化：如果词里没有 p0，肯定不需要合并，直接跳过计算
        if p0 not in ids:
            new_vocab[ids] = freq
            continue

        new_ids = []
        i = 0
        while i < len(ids):
            # 检查是否匹配 p0, p1
            if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(new_token_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        new_vocab[tuple(new_ids)] = freq

    return new_vocab


def train_bpe_parallel(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str] | None = None,
    num_workers: int = 4,
):
    """
    BPE 训练主入口
    """
    print(f"--- Starting BPE Training on {input_path} ---")

    # 1. 计算分块边界
    # 假设特殊 token 是 <|endoftext|>，如果不包含，可以传空 bytes 或其他
    special_tokens = special_tokens or []
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b""
    boundaries = find_chunk_boundaries(input_path, num_workers, split_token)

    # 准备参数 [(file, start, end), (file, start, end), ...]
    chunk_args = []
    for i in range(len(boundaries) - 1):
        chunk_args.append((input_path, boundaries[i], boundaries[i + 1], special_tokens))

    print(f"Divided file into {len(chunk_args)} chunks. Running pre-tokenization...")

    # 2. 并行 Pre-tokenization (Map 阶段)
    # 使用 multiprocessing.Pool 自动管理进程
    global_vocab_counts = collections.Counter()
    if num_workers > 1 and len(chunk_args) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap_unordered 稍微快一点，因为我们不关心顺序
            for local_counts in pool.imap_unordered(_process_chunk_worker, chunk_args):
                global_vocab_counts.update(local_counts)
    else:
        for args in chunk_args:
            global_vocab_counts.update(_process_chunk_worker(args))

    print(f"Pre-tokenization complete. Unique words: {len(global_vocab_counts)}")

    # 3. BPE 迭代 (Serial Merge 阶段)
    # 初始 token 0-255
    merges = []
    # 从 256 开始分配新 ID (假设是 byte-level BPE)
    current_token_id = 256
    base_vocab_size = 256 + len(special_tokens)
    target_merges = vocab_size - base_vocab_size
    if target_merges < 0:
        raise ValueError("vocab_size too small for base vocabulary + special tokens")

    # 将 Counter 转为普通 dict 以便处理，虽然 Counter 也能用但 dict 更轻量
    vocab_counts = dict(global_vocab_counts)

    id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    print("Starting Merge Iterations...")

    for i in range(target_merges):
        # a. 统计 Pair 频率
        pairs = get_stats(vocab_counts)

        if not pairs:
            print("No more pairs to merge. Stopping early.")
            break

        # b. 找到频率最高的 Pair
        # 按照 (频率, 字节序) 排序，保证确定性
        best_pair = max(
            pairs,
            key=lambda p: (
                pairs[p],
                id_to_bytes[p[0]],
                id_to_bytes[p[1]],
            ),
        )

        # c. 记录合并规则
        merges.append((best_pair, current_token_id))
        id_to_bytes[current_token_id] = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]

        if i % 100 == 0:
            print(f"Merge {i + 1}/{target_merges}: {best_pair} -> {current_token_id} (freq: {pairs[best_pair]})")

        # d. 更新词表
        # 这一步是单线程的，但因为是在 len(vocab) 上操作，通常很快
        vocab_counts = merge_vocab(best_pair, vocab_counts, current_token_id)

        current_token_id += 1
    longest_token = max(id_to_bytes.values(), key=len)
    print(f"Longest token length after training: {len(longest_token)} bytes")
    print("Training Complete.")
    return merges, vocab_counts


def _build_vocab_and_merges(
    merges_with_ids: List[Tuple[Tuple[int, int], int]],
    special_tokens: List[str] | None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    for (p0, p1), new_id in merges_with_ids:
        b0 = id_to_bytes[p0]
        b1 = id_to_bytes[p1]
        merges.append((b0, b1))
        id_to_bytes[new_id] = b0 + b1

    special_tokens = special_tokens or []
    vocab: dict[int, bytes] = {}
    next_id = 0
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    for i in range(256):
        vocab[next_id] = id_to_bytes[i]
        next_id += 1
    for token_id in range(256, 256 + len(merges_with_ids)):
        vocab[next_id] = id_to_bytes[token_id]
        next_id += 1

    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    if rust_train_bpe is not None:
        num_threads = kwargs.get("num_workers", 0)
        if rust_train_bpe_from_buffer is not None:
            try:
                import mmap

                if os.path.getsize(input_path) > 0:
                    with open(input_path, "rb") as f:
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            vocab_items, merges = rust_train_bpe_from_buffer(
                                mm,
                                vocab_size,
                                special_tokens,
                                num_threads,
                            )
                else:
                    vocab_items, merges = rust_train_bpe(str(input_path), vocab_size, special_tokens, num_threads)
            except Exception:
                vocab_items, merges = rust_train_bpe(str(input_path), vocab_size, special_tokens, num_threads)
        else:
            vocab_items, merges = rust_train_bpe(str(input_path), vocab_size, special_tokens, num_threads)
        vocab = {token_id: bytes(token_bytes) for token_id, token_bytes in vocab_items}
        merges = [(bytes(a), bytes(b)) for a, b in merges]
        return vocab, merges

    merges_with_ids, _ = train_bpe_parallel(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
    return _build_vocab_and_merges(merges_with_ids, special_tokens)

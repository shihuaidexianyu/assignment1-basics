use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;

use onig::Regex;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use rayon::prelude::*;

fn split_non_special_segments<'a>(text: &'a str, special_tokens: &[String]) -> Vec<&'a str> {
    if special_tokens.is_empty() {
        return vec![text];
    }
    let mut tokens: Vec<&str> = special_tokens.iter().map(|s| s.as_str()).collect();
    tokens.sort_by(|a, b| b.len().cmp(&a.len()));

    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut positions: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    positions.push(text.len());
    let mut idx_pos = 0usize;

    while idx_pos + 1 < positions.len() {
        let i = positions[idx_pos];
        let slice = &text[i..];
        let mut matched: Option<usize> = None;
        for token in &tokens {
            if slice.starts_with(token) {
                matched = Some(token.len());
                break;
            }
        }
        if let Some(len) = matched {
            if start < i {
                segments.push(&text[start..i]);
            }
            let new_i = i + len;
            start = new_i;
            while idx_pos + 1 < positions.len() && positions[idx_pos] < new_i {
                idx_pos += 1;
            }
            continue;
        }
        idx_pos += 1;
    }

    if start < text.len() {
        segments.push(&text[start..]);
    }
    segments
}

fn text_from_bytes(bytes: &[u8]) -> Cow<'_, str> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Cow::Borrowed(text),
        Err(_) => Cow::Owned(decode_utf8_ignore(bytes)),
    }
}

fn buffer_as_bytes<'py>(py: Python<'py>, buffer: &PyBuffer<u8>) -> PyResult<&'py [u8]> {
    let slice = buffer.as_slice(py).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "buffer is not C-contiguous; cannot provide zero-copy view",
        )
    })?;
    let ptr = slice.as_ptr() as *const u8;
    let len = slice.len();
    // SAFETY: PyBuffer enforces lifetime tied to Python GIL and we only read.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn pretokenize_and_count_impl(text: &str, special_tokens: &[String]) -> PyResult<Vec<(Vec<u8>, u64)>> {
    let pat = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let re = Regex::new(pat).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
    let segments = split_non_special_segments(text, special_tokens);

    for segment in segments {
        for m in re.find_iter(segment) {
            let token = &segment[m.0..m.1];
            let entry = counts.entry(token.as_bytes().to_vec()).or_insert(0);
            *entry += 1;
        }
    }

    Ok(counts.into_iter().collect())
}

#[pyfunction]
fn pretokenize_and_count(text: &str, special_tokens: Vec<String>) -> PyResult<Vec<(Vec<u8>, u64)>> {
    pretokenize_and_count_impl(text, &special_tokens)
}

#[pyfunction]
fn pretokenize_and_count_from_buffer(
    py: Python<'_>,
    data: PyBuffer<u8>,
    special_tokens: Vec<String>,
) -> PyResult<Vec<(Vec<u8>, u64)>> {
    let bytes = buffer_as_bytes(py, &data)?;
    let text = text_from_bytes(bytes);
    pretokenize_and_count_impl(text.as_ref(), &special_tokens)
}

fn merge_ids(ids: &[u32], p0: u32, p1: u32, new_id: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(ids.len());
    let mut i = 0usize;
    let mut changed = false;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == p0 && ids[i + 1] == p1 {
            merged.push(new_id);
            i += 2;
            changed = true;
        } else {
            merged.push(ids[i]);
            i += 1;
        }
    }
    if changed {
        merged
    } else {
        ids.to_vec()
    }
}

fn decode_utf8_ignore(bytes: &[u8]) -> String {
    let lossy = String::from_utf8_lossy(bytes);
    lossy.chars().filter(|c| *c != '\u{FFFD}').collect()
}

fn lex_compare(a: &[u8], b: &[u8]) -> Ordering {
    let min_len = a.len().min(b.len());
    for i in 0..min_len {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    a.len().cmp(&b.len())
}

fn collect_token_counts(
    text: &str,
    special_tokens: &[String],
    num_threads: usize,
) -> PyResult<HashMap<Vec<u8>, u64>> {
    let pat = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let re = Regex::new(pat).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let segments = split_non_special_segments(text, special_tokens);
    let collect_counts = || {
        segments
            .par_iter()
            .map(|segment| {
                let mut local_counts: HashMap<Vec<u8>, u64> = HashMap::new();
                for m in re.find_iter(segment) {
                    let token = &segment[m.0..m.1];
                    let entry = local_counts.entry(token.as_bytes().to_vec()).or_insert(0);
                    *entry += 1;
                }
                local_counts
            })
            .reduce(HashMap::new, |mut acc, local| {
                for (k, v) in local {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            })
    };

    if num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(pool.install(collect_counts))
    } else {
        Ok(collect_counts())
    }
}

fn train_bpe_impl(
    text: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
    num_threads: usize,
) -> PyResult<(Vec<(u32, Vec<u8>)>, Vec<(Vec<u8>, Vec<u8>)>)> {
    let token_counts = collect_token_counts(text, &special_tokens, num_threads)?;

    let base_vocab_size = 256 + special_tokens.len();
    if vocab_size < base_vocab_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "vocab_size too small for base vocabulary + special tokens",
        ));
    }
    let target_merges = vocab_size - base_vocab_size;

    let mut vocab_counts: Vec<(Vec<u32>, u64)> = Vec::with_capacity(token_counts.len());
    for (token_bytes, count) in token_counts {
        let ids: Vec<u32> = token_bytes.iter().map(|b| *b as u32).collect();
        vocab_counts.push((ids, count));
    }

    let mut id_to_bytes: Vec<Vec<u8>> = (0u32..=255u32).map(|b| vec![b as u8]).collect();
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(target_merges);

    for _ in 0..target_merges {
        let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
        for (ids, freq) in vocab_counts.iter() {
            if ids.len() < 2 {
                continue;
            }
            for i in 0..ids.len() - 1 {
                let entry = pair_counts.entry((ids[i], ids[i + 1])).or_insert(0);
                *entry += *freq;
            }
        }
        if pair_counts.is_empty() {
            break;
        }

        let mut best_pair: Option<(u32, u32)> = None;
        let mut best_count: u64 = 0;
        for (pair, count) in pair_counts.into_iter() {
            let should_take = if best_pair.is_none() {
                true
            } else if count > best_count {
                true
            } else if count == best_count {
                let (bp0, bp1) = best_pair.unwrap();
                let cmp0 = lex_compare(&id_to_bytes[pair.0 as usize], &id_to_bytes[bp0 as usize]);
                if cmp0 == Ordering::Greater {
                    true
                } else if cmp0 == Ordering::Equal {
                    let cmp1 = lex_compare(&id_to_bytes[pair.1 as usize], &id_to_bytes[bp1 as usize]);
                    cmp1 == Ordering::Greater
                } else {
                    false
                }
            } else {
                false
            };

            if should_take {
                best_pair = Some(pair);
                best_count = count;
            }
        }

        let (p0, p1) = match best_pair {
            Some(pair) => pair,
            None => break,
        };

        let b0 = id_to_bytes[p0 as usize].clone();
        let b1 = id_to_bytes[p1 as usize].clone();
        merges.push((b0.clone(), b1.clone()));

        let new_id = id_to_bytes.len() as u32;
        let mut merged_bytes = b0;
        merged_bytes.extend_from_slice(&b1);
        id_to_bytes.push(merged_bytes);

        let mut new_vocab_counts: Vec<(Vec<u32>, u64)> = Vec::with_capacity(vocab_counts.len());
        for (ids, freq) in vocab_counts.into_iter() {
            let merged_ids = merge_ids(&ids, p0, p1, new_id);
            new_vocab_counts.push((merged_ids, freq));
        }
        vocab_counts = new_vocab_counts;
    }

    let mut vocab_items: Vec<(u32, Vec<u8>)> = Vec::with_capacity(id_to_bytes.len() + special_tokens.len());
    let mut next_id: u32 = 0;
    for token in &special_tokens {
        vocab_items.push((next_id, token.as_bytes().to_vec()));
        next_id += 1;
    }
    for i in 0..256u32 {
        vocab_items.push((next_id, vec![i as u8]));
        next_id += 1;
    }
    for i in 256u32..id_to_bytes.len() as u32 {
        vocab_items.push((next_id, id_to_bytes[i as usize].clone()));
        next_id += 1;
    }

    Ok((vocab_items, merges))
}

#[pyfunction]
#[pyo3(signature = (input_path, vocab_size, special_tokens, num_threads=0))]
fn train_bpe(
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
    num_threads: usize,
) -> PyResult<(Vec<(u32, Vec<u8>)>, Vec<(Vec<u8>, Vec<u8>)>)> {
    let data = fs::read(input_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let text = text_from_bytes(&data);
    train_bpe_impl(text.as_ref(), vocab_size, special_tokens, num_threads)
}

#[pyfunction]
#[pyo3(signature = (data, vocab_size, special_tokens, num_threads=0))]
fn train_bpe_from_buffer(
    py: Python<'_>,
    data: PyBuffer<u8>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    num_threads: usize,
) -> PyResult<(Vec<(u32, Vec<u8>)>, Vec<(Vec<u8>, Vec<u8>)>)> {
    let bytes = buffer_as_bytes(py, &data)?;
    let text = text_from_bytes(bytes);
    train_bpe_impl(text.as_ref(), vocab_size, special_tokens, num_threads)
}

#[pymodule]
fn cs336_bpe_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pretokenize_and_count, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize_and_count_from_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe_from_buffer, m)?)?;
    Ok(())
}

<!-- Copilot / AI agent guidance for contributors working in this repository -->

# 快速上手 — 给 AI 代理的简明指引

下面的说明以帮助 AI 代码代理（或新贡献者）在此代码库中快速完成常见任务：运行/调试测试、理解主要模块、以及识别修改点。

- 主要语言/环境：Python 3.11+（`pyproject.toml` 指定）。使用 `uv` 管理/激活环境。
- 运行测试（推荐，且是常见开发循环）：
  - 使用 repo 的环境包装器：
    - `uv run pytest`  （也可以用于运行单个测试文件或测试用例，例如 `uv run pytest tests/test_tokenizer.py`）

- 关键目录与文件：
  - `cs336_basics/` — 主要代码（包含 `train_bpe.py`, `pretokenization_example.py` 等）。
  - `tests/` — 单元测试、fixture 和 snapshot（`_snapshots/`）。测试对 API/函数签名有明确期望，改动接口时请先查看测试。
  - `tests/fixtures/` — 测试使用的参考输入（vocab、sample 文本、merges 等）。
  - `data/data/` — 用于训练/实验的示例数据（存储外部下载的数据）。
  - `make_submission.sh` — (课程相关) 提交脚本；修改实现前后可用来检查提交格式。
  - `pyproject.toml` — 列出了依赖（例如 `torch`, `pytest`, `tiktoken`）和 `uv` 构建后端。

- 项目“心智模型”（大局、架构与数据流）
  - 这是一个教学作业仓库，围绕“实现基础组件以通过测试”构建；测试专门检查函数实现（例如 adapters 里定义的抽象）。
  - BPE 训练流程示例：`cs336_basics/train_bpe.py` 中实现了将大文件切分为独立块（按 special token b"<|endoftext|>" 搜索边界），预处理每个 chunk，然后累积 pre-token 频次以训练 vocab。若你改动分块逻辑或 special token，务必相应更新测试/fixtures。

- 项目约定与注意事项（对 AI 代理尤其有用）
  - 测试先于实现：README 提到最初测试会报 `NotImplementedError`，并且 `tests/adapters.py` 用作连接点——在实现任务时优先查看该文件以确保接口契合。
  - Snapshot tests：`tests/_snapshots/` 包含二进制/NPZ 参考文件。若你修改数值计算实现或 rng，可能需要更新 snapshot（谨慎）。
  - 编码/IO：在 `train_bpe.py` 里对二进制文件分块后再 decode 为 UTF-8（在切分与 special token 查找上要注意以字节为单位的边界处理）。示例 special token 为 `b"<|endoftext|>"`。
  - 小脚本示例：`cs336_basics/pretokenization_example.py` 是演示预分词的参考实现，阅读它可以快速理解预处理约定。

- 常用命令示例（直接可复制）
  - 运行所有测试（推荐）：
    - `uv run pytest`
  - 运行单个测试文件：
    - `uv run pytest tests/test_tokenizer.py`
  - 交互地运行脚本示例：
    - `uv run cs336_basics/pretokenization_example.py`

- 编辑/调试建议
  - 修改实现后先运行受影响的测试文件；如果改动影响数值输出，先检查 `tests/_snapshots/`，是否需要更新快照。
  - 若新增依赖或需要本地调试交互，`uv` 会按 `pyproject.toml` 解决并激活环境，优先用 `uv run` 来保证一致环境。

如果这里有任何缺失（例如你想让 AI 增加特定的代码生成风格、注释规范或测试写法示例），请指出具体需求，我会把说明合并进本文件。

# -*- coding: utf-8 -*-
import os
import re
from typing import List, Dict
from loguru import logger


CHUNK_SIZE = 800       # 目标 chunk 字符数
CHUNK_OVERLAP = 1      # 保留前一个完整段落作为语义桥接
HARD_LIMIT = 1100      # 单块硬上限，超出后按句子继续切


class DocumentLoader:
    """
    文档加载与分块模块
    支持格式：.txt .md .pdf .docx

    分块策略：
    1. 优先按标题/空行切成结构块
    2. 结构块聚合为 chunk，尽量保持段落连续
    3. 超长段落再按句子切分
    4. 新 chunk 自动携带上一 chunk 的末段，避免语义断裂
    """

    def load_dir(self, dir_path: str) -> List[Dict]:
        docs = []
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            return docs

        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                file_docs = self.load_file(fpath)
                docs.extend(file_docs)
                logger.info(f"Loaded: {fname} ({len(file_docs)} chunks)")
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")

        logger.info(f"Total loaded: {len(docs)} chunks from {dir_path}")
        return docs

    def load_file(self, file_path: str) -> List[Dict]:
        ext = os.path.splitext(file_path)[1].lower()
        fname = os.path.basename(file_path)

        if ext in ('.txt', '.md'):
            content = self._load_text(file_path)
        elif ext == '.pdf':
            content = self._load_pdf(file_path)
        elif ext == '.docx':
            content = self._load_docx(file_path)
        else:
            logger.warning(f"Unsupported format: {ext}")
            return []

        if not content.strip():
            return []

        chunks = self._chunk(content)
        total = len(chunks)
        return [
            {
                "doc_id": f"{fname}_chunk_{i}",
                "content": chunk,
                "metadata": {
                    "source": file_path,
                    "filename": fname,
                    "chunk_index": i,
                    "total_chunks": total,
                    "prev_chunk_id": f"{fname}_chunk_{i-1}" if i > 0 else None,
                    "next_chunk_id": f"{fname}_chunk_{i+1}" if i < total - 1 else None,
                    "section_path": f"{fname} [chunk {i+1}/{total}]"
                }
            }
            for i, chunk in enumerate(chunks)
        ]

    def _load_text(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _load_pdf(self, path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            return text
        except ImportError:
            logger.error("pypdf not installed: pip install pypdf")
            return ''
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return ''

    def _load_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            logger.error("python-docx not installed: pip install python-docx")
            return ''
        except Exception as e:
            logger.error(f"DOCX load failed: {e}")
            return ''

    def _chunk(self, text: str) -> List[str]:
        """
        结构感知分块：优先保持标题、段落、句子连续性。
        """
        blocks = self._split_into_blocks(text)
        if not blocks:
            cleaned = text.strip()
            return [cleaned] if cleaned else []

        chunks: List[str] = []
        current_blocks: List[str] = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if len(block) > HARD_LIMIT:
                sub_blocks = self._split_long_block(block)
            else:
                sub_blocks = [block]

            for sub in sub_blocks:
                candidate = self._join_blocks(current_blocks + [sub])
                if current_blocks and len(candidate) > CHUNK_SIZE:
                    chunks.append(self._join_blocks(current_blocks))
                    overlap_blocks = current_blocks[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else []
                    current_blocks = overlap_blocks + [sub]

                    if len(self._join_blocks(current_blocks)) > HARD_LIMIT:
                        chunks.append(self._join_blocks(current_blocks))
                        current_blocks = []
                else:
                    current_blocks.append(sub)

        if current_blocks:
            chunks.append(self._join_blocks(current_blocks))

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_into_blocks(self, text: str) -> List[str]:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        blocks: List[str] = []
        current: List[str] = []
        in_code_block = False

        for line in lines:
            raw = line.rstrip()
            stripped = raw.strip()

            if stripped.startswith("```"):
                if current:
                    blocks.append('\n'.join(current).strip())
                    current = []
                in_code_block = not in_code_block
                current.append(raw)
                continue

            if in_code_block:
                current.append(raw)
                continue

            is_heading = bool(re.match(r'^(#{1,6})\s+', stripped))
            is_blank = stripped == ''

            if is_heading:
                if current:
                    blocks.append('\n'.join(current).strip())
                    current = []
                current.append(raw)
                continue

            if is_blank:
                if current:
                    blocks.append('\n'.join(current).strip())
                    current = []
                continue

            current.append(raw)

        if current:
            blocks.append('\n'.join(current).strip())

        return [b for b in blocks if b]

    def _split_long_block(self, block: str) -> List[str]:
        sentences = self._split_sentences(block)
        if len(sentences) <= 1:
            return [block]

        parts: List[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current}{sentence}" if current else sentence
            if current and len(candidate) > CHUNK_SIZE:
                parts.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current.strip():
            parts.append(current.strip())
        return parts if parts else [block]

    def _split_sentences(self, text: str) -> List[str]:
        pieces = re.split(r'(?<=[。！？；.!?;])\s*', text)
        return [p for p in pieces if p and p.strip()]

    def _join_blocks(self, blocks: List[str]) -> str:
        return '\n\n'.join(block.strip() for block in blocks if block and block.strip())

# -*- coding: utf-8 -*-
import os
from typing import List, Dict
from loguru import logger


CHUNK_SIZE = 500      # 每个 chunk 的字符数
CHUNK_OVERLAP = 50   # chunk 之间的重叠字符数


class DocumentLoader:
    """
    文档加载与分块模块
    支持格式：.txt .md .pdf .docx
    """

    def load_dir(self, dir_path: str) -> List[Dict]:
        """
        加载目录下所有支持格式的文档
        """
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
        """
        加载单个文件并分块
        """
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
        return [
            {
                "doc_id": f"{fname}_chunk_{i}",
                "content": chunk,
                "metadata": {
                    "source": file_path,
                    "filename": fname,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "section_path": f"{fname} [chunk {i+1}/{len(chunks)}]"
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
            text = '\n'.join(
                page.extract_text() or '' for page in reader.pages
            )
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
        滑动窗口分块
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks if chunks else [text.strip()]

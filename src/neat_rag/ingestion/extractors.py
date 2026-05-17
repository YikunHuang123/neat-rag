import time
from pathlib import Path
from typing import Dict, Any, Tuple

from neat_rag.exceptions import ExtractionError, UnsupportedFileTypeError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


class PdfExtractor:
    """Extracts text and structure from PDF files using Docling."""

    def __init__(
        self,
        enable_ocr: bool = False,
        include_images: bool = True,
        include_tables: bool = True,
        images_scale: float = 2.0,
    ):
        self._enable_ocr = enable_ocr
        self._include_images = include_images
        self._include_tables = include_tables
        self._images_scale = images_scale
        self._converter = None  # lazy init — docling is slow to import

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            opts = PdfPipelineOptions()
            opts.do_ocr = self._enable_ocr
            opts.do_picture_description = self._include_images
            opts.do_table_structure = self._include_tables
            opts.images_scale = self._images_scale

            if self._include_images or self._enable_ocr:
                logger.info(
                    "🚀 Initializing Docling models... If this is the first run, it may download ~1GB of weights (SmolVLM/OCR). "
                    "This might take several minutes depending on your network speed.",
                    include_images=self._include_images,
                    enable_ocr=self._enable_ocr
                )

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        return self._converter

    def extract(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Return (markdown_text, metadata) for a PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            t0 = time.time()
            result = self._get_converter().convert(str(file_path))
            elapsed = round(time.time() - t0, 2)
            doc = result.document
            content = doc.export_to_markdown()
            metadata = {
                "title": file_path.name,
                "source": str(file_path),
                "mime_type": "application/pdf",
                "pages": len(doc.pages),
                "pictures": len(doc.pictures),
                "tables": len(doc.tables),
                "processing_time": elapsed,
                "extraction_method": "docling",
            }
            logger.info("Extracted PDF", file=file_path.name, pages=metadata["pages"], elapsed_s=elapsed)
            return content, metadata
        except (FileNotFoundError, ExtractionError):
            raise
        except Exception as e:
            logger.error("PDF extraction failed", file=file_path.name, error=str(e))
            raise ExtractionError(f"Failed to extract PDF '{file_path.name}': {e}") from e


class DocxExtractor:
    """Extracts text from DOCX files using python-docx."""

    def extract(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            import docx
            document = docx.Document(str(file_path))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            content = "\n\n".join(paragraphs)
            metadata = {
                "title": file_path.name,
                "source": str(file_path),
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "extraction_method": "python-docx",
            }
            logger.info("Extracted DOCX", file=file_path.name, paragraphs=len(paragraphs))
            return content, metadata
        except (FileNotFoundError, ExtractionError):
            raise
        except Exception as e:
            logger.error("DOCX extraction failed", file=file_path.name, error=str(e))
            raise ExtractionError(f"Failed to extract DOCX '{file_path.name}': {e}") from e


class MarkdownExtractor:
    """Reads .md files as plain text (markdown formatting preserved)."""

    def extract(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            metadata = {
                "title": file_path.name,
                "source": str(file_path),
                "mime_type": "text/markdown",
                "extraction_method": "plain",
            }
            return content, metadata
        except (FileNotFoundError, ExtractionError):
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to read markdown '{file_path.name}': {e}") from e


class PlainTextExtractor:
    """Reads .txt files as plain text."""

    def extract(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            metadata = {
                "title": file_path.name,
                "source": str(file_path),
                "mime_type": "text/plain",
                "extraction_method": "plain",
            }
            return content, metadata
        except (FileNotFoundError, ExtractionError):
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to read text file '{file_path.name}': {e}") from e


class HtmlExtractor:
    """Extracts readable text from HTML files using trafilatura."""

    def extract(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            import trafilatura
            raw_html = file_path.read_text(encoding="utf-8", errors="replace")
            # trafilatura.extract returns None if it can't parse; fall back to raw html
            content = trafilatura.extract(raw_html) or raw_html
            metadata = {
                "title": file_path.name,
                "source": str(file_path),
                "mime_type": "text/html",
                "extraction_method": "trafilatura",
            }
            logger.info("Extracted HTML", file=file_path.name, content_len=len(content))
            return content, metadata
        except (FileNotFoundError, ExtractionError):
            raise
        except Exception as e:
            logger.error("HTML extraction failed", file=file_path.name, error=str(e))
            raise ExtractionError(f"Failed to extract HTML '{file_path.name}': {e}") from e


# ---------------------------------------------------------------------------
# Dispatch table — maps file extension → extractor class
# ---------------------------------------------------------------------------

_EXT_MAP: Dict[str, type] = {
    ".pdf":  PdfExtractor,
    ".docx": DocxExtractor,
    ".md":   MarkdownExtractor,
    ".txt":  PlainTextExtractor,
    ".html": HtmlExtractor,
    ".htm":  HtmlExtractor,
}

SUPPORTED_EXTENSIONS = set(_EXT_MAP.keys())

# Extractor singletons — heavy extractors (PdfExtractor/Docling) are expensive
# to initialize; reuse instances across requests to avoid reloading models.
_extractor_cache: Dict[str, Any] = {}


def dispatch_by_ext(file_path: Path) -> "PdfExtractor | DocxExtractor | MarkdownExtractor | PlainTextExtractor | HtmlExtractor":
    """Return a cached extractor instance for the given file extension."""
    ext = file_path.suffix.lower()
    extractor_cls = _EXT_MAP.get(ext)
    if extractor_cls is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext not in _extractor_cache:
        _extractor_cache[ext] = extractor_cls()
    return _extractor_cache[ext]

import os.path
import os
from typing import Dict, List, Optional
from pathlib import Path
from fsspec import AbstractFileSystem
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
from llmsherpa.readers import LayoutPDFReader


# Extend the PDFReader class to create a custom reader
class LLMSherapaReader(PDFReader):
    """LLM Sherpa PDF Parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""

        docs = []
        chunk_text = ""
        file_name = os.path.basename(file)

        # Load the PDF
        llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(str(file))
        chunks = doc.chunks()

        for chunk in chunks:
            chunk_text = chunk.to_text()
            page_label = chunk.page_idx
            section_header = chunk.parent_text()
            metadata = { "page_label": page_label,
                         "file_name": file_name, 
                         "section_header": section_header}
            if extra_info is not None:
                metadata.update(extra_info)

            docs.append(Document(text=chunk_text, metadata=metadata))

        return docs
    
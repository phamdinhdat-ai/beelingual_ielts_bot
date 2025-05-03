import os
import json 
import logging 
import pathlib
from pathlib import Path
import yaml
from typing import Any, Dict, List, Tuple, Iterator
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from langchain_core.document_loaders import  BaseLoader 
from langchain_core.documents import Document


class DocumentCustomConverter(BaseLoader):
        
        def __init__(self, filepath: List[str], type_doc:str = 'markdown') -> None: 
            self._filepath = filepath if isinstance(filepath, list) else [filepath]
            self._type_doc = type_doc
            self.pipeline_options = PdfPipelineOptions()
            self.pipeline_options.do_ocr = False 
            self.pipeline_options.do_table_structure = True 
            self.document_coverter = (
                DocumentConverter(
                    allowed_formats=[
                        InputFormat.PDF,
                        InputFormat.DOCX,
                        InputFormat.HTML,
                        InputFormat.PPTX, 
                        InputFormat.XLSX
                    ], 
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options,
                                                                    backend=PyPdfiumDocumentBackend), 
                                    InputFormat.DOCX: WordFormatOption(pipeline_cls = SimplePipeline)} 
                )
            )
            
        def lazy_load(self) -> Iterator[Document]:
            for file in self._filepath:
                coverted_docment  = self.document_coverter.convert(file , raises_on_error=False)
                if self._type_doc == 'markdown':
                    yield Document(page_content=coverted_docment.document.export_to_markdown())
                else:
                    yield Document(page_content=coverted_docment.document.export_to_dict())


def covert_document(file_path:str, type_doc:str = 'markdown'):
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False 
    pipeline_options.do_table_structure = True 
    document_coverter = (
        DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX, 
                InputFormat.XLSX
            ], 
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options,
                                                             backend=PyPdfiumDocumentBackend), 
                            InputFormat.DOCX: WordFormatOption(pipeline_cls = SimplePipeline)} 
        )
    )
    
    coverted_docment  = document_coverter.convert(file_path , raises_on_error=False)
    if type_doc == 'markdown':
        return coverted_docment.document.export_to_markdown()
    else:
        return coverted_docment.document.export_to_dict()
    
    


def test_convert_document(file_path, type_doc):
    document_converter  = DocumentCustomConverter(filepath=file_path, type_doc=type_doc)
    documents = document_converter.lazy_load()
    return documents

# file_path = "backend/data/2501.07329v2.pdf"
# documents = covert_document(file_path=file_path)
# print(documents)
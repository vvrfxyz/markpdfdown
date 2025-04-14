from core.dataset import PymuDocDataset
from core.doc_analyze import doc_analyze

pdf_file_path = "/data/markpdfdown/PMC11742413_pages_1-3.pdf"
# pdf_bytes = ""
with open(pdf_file_path, 'rb') as f:
    pdf_bytes = f.read()

ds = PymuDocDataset(pdf_bytes)
result = ds.apply(doc_analyze)

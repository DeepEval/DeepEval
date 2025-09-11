import PyPDF2

class PDFHandler:
    """
    The class allows merging multiple PDF files into one and extracting text from PDFs using PyPDF2 library.
    """

    def __init__(self, filepaths):
        """
        takes a list of file paths filepaths as a parameter.
        It creates a list named readers using PyPDF2, where each reader opens a file from the given paths.
        """
        self.filepaths = filepaths
        self.readers = [PyPDF2.PdfFileReader(fp) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        """
        Read files in self.readers which stores handles to multiple PDF files.
        Merge them to one pdf and update the page number, then save in disk.
        :param output_filepath: str, ouput file path to save to
        :return: str, "Merged PDFs saved at {output_filepath}" if successfully merged
        """
        merger = PyPDF2.PdfFileMerger()
        for reader in self.readers:
            for page in range(reader.numPages):
                merger.append(reader, pages=page)
        merger.write(output_filepath)
        merger.close()
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        """
        Extract text from pdf files in self.readers
        :return pdf_texts: list of str, each element is the text of one pdf file
        >>> handler = PDFHandler(['a.pdf', 'b.pdf'])
        >>> handler.extract_text_from_pdfs()
        ['Test a.pdf', 'Test b.pdf']
        """
        pdf_texts = [self.readers[i].getPage(i).extractText() for i in range(len(self.readers))]
        return pdf_texts

if __name__ == "__main__":
    # Test case for merge_pdfs
    handler = PDFHandler(['a.pdf', 'b.pdf'])
    output = handler.merge_pdfs('out.pdf')
    print(output)

    # Test case for extract_text_from_pdfs
    handler = PDFHandler(['a.pdf', 'b.pdf'])
    texts = handler.extract_text_from_pdfs()
    for text in texts:
        print(text)
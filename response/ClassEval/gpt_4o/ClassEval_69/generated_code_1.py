import PyPDF2

class PDFHandler:
    """
    The class allows merging multiple PDF files into one and extracting text from PDFs using PyPDF2 library.
    """

    def __init__(self, filepaths):
        """
        Takes a list of file paths filepaths as a parameter.
        It creates a list named readers using PyPDF2, where each reader opens a file from the given paths.
        """
        self.filepaths = filepaths
        self.readers = [PyPDF2.PdfFileReader(open(fp, 'rb')) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        """
        Read files in self.readers which stores handles to multiple PDF files.
        Merge them to one PDF and update the page number, then save it to disk.
        :param output_filepath: str, output file path to save to
        :return: str, "Merged PDFs saved at {output_filepath}" if successfully merged
        """
        merger = PyPDF2.PdfMerger()
        for reader in self.readers:
            merger.append(reader)
        with open(output_filepath, 'wb') as output_file:
            merger.write(output_file)
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        """
        Extract text from PDF files in self.readers.
        :return pdf_texts: list of str, each element is the text of one PDF file
        """
        pdf_texts = []
        for reader in self.readers:
            text = ""
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()  # Updated method name in PyPDF2
            pdf_texts.append(text)
        return pdf_texts

if __name__ == "__main__":
    # Test case for merging PDFs
    handler = PDFHandler(['a.pdf', 'b.pdf'])
    merge_output = handler.merge_pdfs('out.pdf')
    print(merge_output)  # Expected: "Merged PDFs saved at out.pdf"

    # Test case for extracting text from PDFs
    extract_output = handler.extract_text_from_pdfs()
    print(extract_output)  # Expected output: ['Text from a.pdf', 'Text from b.pdf']
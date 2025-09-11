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
        Reads files in self.readers which stores handles to multiple PDF files.
        Merges them to one PDF and saves it to disk.
        :param output_filepath: str, output file path to save to
        :return: str, "Merged PDFs saved at {output_filepath}" if successfully merged
        """
        writer = PyPDF2.PdfFileWriter()
        for reader in self.readers:
            for page_num in range(reader.getNumPages()):
                writer.addPage(reader.getPage(page_num))
        with open(output_filepath, 'wb') as output_file:
            writer.write(output_file)
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        """
        Extracts text from PDF files in self.readers
        :return pdf_texts: list of str, each element is the text of one PDF file
        """
        pdf_texts = []
        for reader in self.readers:
            page_texts = []
            for page_num in range(reader.getNumPages()):
                page = reader.getPage(page_num)
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
            pdf_texts.append('\n'.join(page_texts))
        return pdf_texts

# Test cases
if __name__ == "__main__":
    # Test merge_pdfs
    handler = PDFHandler(['a.pdf', 'b.pdf'])
    output = handler.merge_pdfs('out.pdf')
    print(output)

    # Test extract_text_from_pdfs
    handler = PDFHandler(['a.pdf', 'b.pdf'])
    texts = handler.extract_text_from_pdfs()
    for text in texts:
        print(text)
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
        self.readers = [PyPDF2.PdfReader(fp) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        """
        Read files in self.readers which stores handles to multiple PDF files.
        Merge them into one PDF and save it to disk.
        :param output_filepath: str, output file path to save to
        :return: str, "Merged PDFs saved at {output_filepath}" if successfully merged
        """
        pdf_writer = PyPDF2.PdfWriter()
        
        for reader in self.readers:
            for page in range(len(reader.pages)):
                pdf_writer.add_page(reader.pages[page])

        with open(output_filepath, 'wb') as out_file:
            pdf_writer.write(out_file)

        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        """
        Extract text from pdf files in self.readers
        :return pdf_texts: list of str, each element is the text of one pdf file
        """
        pdf_texts = []
        
        for reader in self.readers:
            text = ''
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text() or ''
            pdf_texts.append(text.strip())
        
        return pdf_texts

# Test cases
if __name__ == "__main__":
    # Note: Ensure 'a.pdf' and 'b.pdf' files exist in the working directory for testing.
    
    # Test for merging PDFs
    merge_handler = PDFHandler(['a.pdf', 'b.pdf'])
    output_merge = merge_handler.merge_pdfs('merged_output.pdf')
    print(output_merge)

    # Test for extracting text from PDFs
    extract_handler = PDFHandler(['a.pdf', 'b.pdf'])
    output_texts = extract_handler.extract_text_from_pdfs()
    print(output_texts)
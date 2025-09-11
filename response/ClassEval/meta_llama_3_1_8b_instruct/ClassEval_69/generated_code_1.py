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
        >>> handler = PDFHandler(['a.pdf', 'b.pdf'])
        >>> handler.merge_pdfs('out.pdf')
        Merged PDFs saved at out.pdf
        """
        # Create a writer object
        output_pdf = PyPDF2.PdfFileWriter()

        # Iterate through each reader in self.readers
        for reader in self.readers:
            # Get the number of pages in the current reader
            num_pages = reader.numPages
            # Iterate through each page in the reader
            for page in range(num_pages):
                # Add the page to the output pdf
                output_pdf.addPage(reader.getPage(page))

        # Save the output pdf
        with open(output_filepath, "wb") as output:
            output_pdf.write(output)

        # Return a success message
        return f"Merged PDFs saved at {output_filepath}"

    def extract_text_from_pdfs(self):
        """
        Extract text from pdf files in self.readers
        :return pdf_texts: list of str, each element is the text of one pdf file
        >>> handler = PDFHandler(['a.pdf', 'b.pdf'])
        >>> handler.extract_text_from_pdfs()
        ['Test a.pdf', 'Test b.pdf']
        """
        # Create an empty list to store the extracted text
        pdf_texts = []

        # Iterate through each reader in self.readers
        for reader in self.readers:
            # Create a string to store the text of the current reader
            text = ""

            # Get the number of pages in the current reader
            num_pages = reader.numPages

            # Iterate through each page in the reader
            for page in range(num_pages):
                # Extract the text from the current page and add it to the text string
                text += reader.getPage(page).extractText()

            # Add the text of the current reader to the pdf_texts list
            pdf_texts.append(text)

        # Return the list of extracted text
        return pdf_texts

if __name__ == "__main__":
    # Test case for merge_pdfs method
    handler = PDFHandler(['test1.pdf', 'test2.pdf'])
    print(handler.merge_pdfs('output.pdf'))

    # Test case for extract_text_from_pdfs method
    handler = PDFHandler(['test1.pdf', 'test2.pdf'])
    print(handler.extract_text_from_pdfs())
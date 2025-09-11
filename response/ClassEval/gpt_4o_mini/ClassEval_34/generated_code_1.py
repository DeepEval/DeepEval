from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


class DocFileHandler:
    """
    This is a class that handles Word documents and provides functionalities for reading, writing, and modifying the content of Word documents.
    """

    def __init__(self, file_path):
        """
        Initializes the DocFileHandler object with the specified file path.
        :param file_path: str, the path to the Word document file.
        """
        self.file_path = file_path
        self.document = Document()

    def read_text(self):
        """
        Reads the content of a Word document and returns it as a string.
        :return: str, the content of the Word document.
        """
        try:
            document = Document(self.file_path)
            content = []
            for para in document.paragraphs:
                content.append(para.text)
            return "\n".join(content)
        except Exception as e:
            return str(e)

    def write_text(self, content, font_size=12, alignment='left'):
        """
        Writes the specified content to a Word document.
        :param content: str, the text content to write.
        :param font_size: int, optional, the font size of the text (default is 12).
        :param alignment: str, optional, the alignment of the text ('left', 'center', or 'right'; default is 'left').
        :return: bool, True if the write operation is successful, False otherwise.
        """
        try:
            paragraph = self.document.add_paragraph(content)
            run = paragraph.runs[0]
            run.font.size = Pt(font_size)
            paragraph.alignment = self._get_alignment_value(alignment)
            self.document.save(self.file_path)
            return True
        except Exception as e:
            print(f"Error writing text: {e}")
            return False

    def add_heading(self, heading, level=1):
        """
        Adds a heading to the Word document.
        :param heading: str, the text of the heading.
        :param level: int, optional, the level of the heading (1, 2, 3, etc.; default is 1).
        :return: bool, True if the heading is successfully added, False otherwise.
        """
        try:
            self.document.add_heading(heading, level=level)
            self.document.save(self.file_path)
            return True
        except Exception as e:
            print(f"Error adding heading: {e}")
            return False

    def add_table(self, data):
        """
        Adds a table to the Word document with the specified data.
        :param data: list of lists, the data to populate the table.
        :return: bool, True if the table is successfully added, False otherwise.
        """
        try:
            table = self.document.add_table(rows=len(data), cols=len(data[0]))
            for row_idx, row in enumerate(data):
                for col_idx, cell_data in enumerate(row):
                    table.cell(row_idx, col_idx).text = str(cell_data)
            self.document.save(self.file_path)
            return True
        except Exception as e:
            print(f"Error adding table: {e}")
            return False

    def _get_alignment_value(self, alignment):
        """
        Returns the alignment value corresponding to the given alignment string.
        :param alignment: str, the alignment string ('left', 'center', or 'right').
        :return: int, the alignment value.
        """
        if alignment == 'center':
            return WD_PARAGRAPH_ALIGNMENT.CENTER
        elif alignment == 'right':
            return WD_PARAGRAPH_ALIGNMENT.RIGHT
        else:
            return WD_PARAGRAPH_ALIGNMENT.LEFT

# Example test cases
if __name__ == "__main__":
    doc_handler = DocFileHandler("test_document.docx")

    # Test case for writing text
    write_output = doc_handler.write_text("Hello, World!", font_size=14, alignment='center')
    print(f"Write output: {write_output}")

    # Test case for adding a heading
    heading_output = doc_handler.add_heading("This is a Heading", level=1)
    print(f"Heading output: {heading_output}")

    # Test case for adding a table
    table_data = [
        ["Header 1", "Header 2"],
        ["Row 1 Col 1", "Row 1 Col 2"],
        ["Row 2 Col 1", "Row 2 Col 2"]
    ]
    table_output = doc_handler.add_table(table_data)
    print(f"Table output: {table_output}")

    # Test case for reading text
    read_output = doc_handler.read_text()
    print(f"Read output:\n{read_output}")
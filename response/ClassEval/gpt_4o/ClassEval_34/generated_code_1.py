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
        content = []
        for para in self.document.paragraphs:
            content.append(para.text)
        return '\n'.join(content)

    def write_text(self, content, font_size=12, alignment='left'):
        """
        Writes the specified content to a Word document.
        :param content: str, the text content to write.
        :param font_size: int, optional, the font size of the text (default is 12).
        :param alignment: str, optional, the alignment of the text ('left', 'center', or 'right'; default is 'left').
        :return: bool, True if the write operation is successful, False otherwise.
        """
        try:
            para = self.document.add_paragraph(content)
            run = para.runs[0]
            run.font.size = Pt(font_size)
            para.alignment = self._get_alignment_value(alignment)
            self.document.save(self.file_path)
            return True
        except Exception as e:
            print(f"Failed to write text: {e}")
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
            print(f"Failed to add heading: {e}")
            return False

    def add_table(self, data):
        """
        Adds a table to the Word document with the specified data.
        :param data: list of lists, the data to populate the table.
        :return: bool, True if the table is successfully added, False otherwise.
        """
        try:
            table = self.document.add_table(rows=len(data), cols=len(data[0]))
            for row_idx, row_data in enumerate(data):
                row_cells = table.rows[row_idx].cells
                for col_idx, cell_data in enumerate(row_data):
                    row_cells[col_idx].text = str(cell_data)
            self.document.save(self.file_path)
            return True
        except Exception as e:
            print(f"Failed to add table: {e}")
            return False

    def _get_alignment_value(self, alignment):
        """
        Returns the alignment value corresponding to the given alignment string.
        :param alignment: str, the alignment string ('left', 'center', or 'right').
        :return: int, the alignment value.
        """
        alignment_map = {
            'left': WD_PARAGRAPH_ALIGNMENT.LEFT,
            'center': WD_PARAGRAPH_ALIGNMENT.CENTER,
            'right': WD_PARAGRAPH_ALIGNMENT.RIGHT
        }
        return alignment_map.get(alignment, WD_PARAGRAPH_ALIGNMENT.LEFT)

if __name__ == "__main__":
    file_path = "test.docx"
    handler = DocFileHandler(file_path)

    # Test case for write_text
    success = handler.write_text("Hello, World!", font_size=14, alignment='center')
    print("Write Text Success:", success)

    # Test case for read_text
    text = handler.read_text()
    print("Read Text Output:", text)

    # Test case for add_heading
    success = handler.add_heading("Sample Heading", level=1)
    print("Add Heading Success:", success)

    # Test case for add_table
    table_data = [
        ["Name", "Age", "City"],
        ["Alice", 30, "New York"],
        ["Bob", 25, "Los Angeles"]
    ]
    success = handler.add_table(table_data)
    print("Add Table Success:", success)
import re
import string
from bs4 import BeautifulSoup

class HtmlUtil:
    """
    This is a class as util for html, supporting for formatting and extracting code from HTML text, including cleaning up the text and converting certain elements into specific marks.
    """

    def __init__(self):
        """
        Initialize a series of labels
        """
        self.SPACE_MARK = '-SPACE-'
        self.JSON_MARK = '-JSON-'
        self.MARKUP_LANGUAGE_MARK = '-MARKUP_LANGUAGE-'
        self.URL_MARK = '-URL-'
        self.NUMBER_MARK = '-NUMBER-'
        self.TRACE_MARK = '-TRACE-'
        self.COMMAND_MARK = '-COMMAND-'
        self.COMMENT_MARK = '-COMMENT-'
        self.CODE_MARK = '-CODE-'

    @staticmethod
    def __format_line_feed(text):
        """
        Replace consecutive line breaks with a single line break
        :param text: string with consecutive line breaks
        :return: string, replaced text with single line break
        """
        return re.sub(r'\s+', ' ', text)

    def format_line_html_text(self, html_text):
        """
        Get the html text without the code, and add the code tag -CODE- where the code is
        :param html_text: string
        :return: string
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        formatted_text = soup.get_text()
        formatted_text = self.__format_line_feed(formatted_text)
        code_blocks = soup.find_all('code')
        for code_block in code_blocks:
            code_block.string = self.CODE_MARK
        formatted_text = soup.get_text()
        code_blocks = soup.find_all('code')
        for code_block in code_blocks:
            code_block.string = str(code_block.string)
        return formatted_text

    def extract_code_from_html_text(self, html_text):
        """
        Extract codes from the html body
        :param html_text: string, html text
        :return: the list of code
        """
        codes = []
        soup = BeautifulSoup(html_text, 'html.parser')
        code_blocks = soup.find_all('code')
        for code_block in code_blocks:
            codes.append(code_block.string)
        return codes

# Test cases
if __name__ == "__main__":
    htmlutil = HtmlUtil()
    
    # Test format_line_html_text
    html_text = """
    <html>
    <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <pre>print('Hello, world!')</pre>
        <p>Another paragraph.</p>
        <pre><code>for i in range(5):
        print(i)</code></pre>
    </body>
    </html>
    """
    formatted_text = htmlutil.format_line_html_text(html_text)
    print("Formatted Text:")
    print(formatted_text)
    
    # Test extract_code_from_html_text
    codes = htmlutil.extract_code_from_html_text(html_text)
    print("\nExtracted Codes:")
    print(codes)
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
        :return:string, replaced text with single line break
        """
        return re.sub(r'\n+', '\n', text).strip()

    def format_line_html_text(self, html_text):
        """
        get the html text without the code, and add the code tag -CODE- where the code is
        :param html_text:string
        :return:string
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        code_blocks = soup.find_all(['pre', 'code'])
        
        # Extract the code and remove it from the original HTML
        for code_block in code_blocks:
            code = code_block.get_text()
            code_block.replace_with(self.CODE_MARK)  # Replace code block with CODE_MARK

        # Get the cleaned text
        cleaned_text = soup.get_text(separator='\n')
        return self.__format_line_feed(cleaned_text)

    def extract_code_from_html_text(self, html_text):
        """
        extract codes from the html body
        :param html_text: string, html text
        :return: the list of code
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        code_blocks = soup.find_all(['pre', 'code'])
        extracted_codes = [code_block.get_text() for code_block in code_blocks]
        return extracted_codes

# Test Cases
if __name__ == "__main__":
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

    htmlutil = HtmlUtil()

    # Test for format_line_html_text
    formatted_text = htmlutil.format_line_html_text(html_text)
    print("Formatted HTML Text:")
    print(formatted_text)

    # Test for extract_code_from_html_text
    extracted_codes = htmlutil.extract_code_from_html_text(html_text)
    print("Extracted Codes:")
    print(extracted_codes)
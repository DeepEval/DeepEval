import re
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
        return re.sub(r'\n+', '\n', text)

    def format_line_html_text(self, html_text):
        """
        get the html text without the code, and add the code tag -CODE- where the code is
        :param html_text:string
        :return:string
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        for code in soup.find_all(['pre', 'code']):
            code.replace_with(self.CODE_MARK)
        text = soup.get_text()
        return self.__format_line_feed(text).strip()

    def extract_code_from_html_text(self, html_text):
        """
        extract codes from the html body
        :param html_text: string, html text
        :return: the list of code
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        codes = []
        for code in soup.find_all(['pre', 'code']):
            codes.append(code.get_text())
        return codes

if __name__ == "__main__":

    htmlutil = HtmlUtil()

    formatted_text = htmlutil.format_line_html_text("""
    <html>
    <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <pre>print('Hello, world!')</pre>
        <p>Another paragraph.</p>
        <code>for i in range(5):
            print(i)</code>
    </body>
    </html>
    """)
    print("Formatted HTML Text:")
    print(formatted_text)
    print()

    extracted_codes = htmlutil.extract_code_from_html_text("""
    <html>
    <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <pre>print('Hello, world!')</pre>
        <p>Another paragraph.</p>
        <code>for i in range(5):
            print(i)</code>
    </body>
    </html>
    """)
    print("Extracted Codes:")
    print(extracted_codes)
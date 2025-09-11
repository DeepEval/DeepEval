import re
import string
import gensim
from bs4 import BeautifulSoup

class HtmlUtil:
    """
    This is a class for util for html, supporting for formatting and extracting code from HTML text, including cleaning up the text and converting certain elements into specific marks.
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
        return re.sub(r'\s+', self.SPACE_MARK, text)

    def format_line_html_text(self, html_text):
        """
        Format HTML text by adding code tags around pre-formatted code blocks.
        :param html_text: string
        :return: string
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        for pre_block in soup.find_all('pre', class_='code'):
            code = pre_block.text.strip()
            formatted_code = self.__format_line_feed(code)
            formatted_code = f'<pre><code>{formatted_code}</code></pre>'
            pre_block.replace_with(formatted_code)
        return str(soup)

    def extract_code_from_html_text(self, html_text):
        """
        Extract code from HTML text.
        :param html_text: string, HTML text
        :return: list of strings, extracted code
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        code_blocks = []
        for pre_block in soup.find_all('pre', class_='code'):
            code_blocks.append(pre_block.text.strip())
        return code_blocks

# Test cases
if __name__ == "__main__":
    html_util = HtmlUtil()
    
    # Test case 1
    input_html = """
    <html>
    <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <pre class="code">print('Hello, world!')</pre>
        <p>Another paragraph.</p>
        <pre class="code"><code>for i in range(5):
            print(i)</code></pre>
        </body>
    </html>
    """
    formatted_html = html_util.format_line_html_text(input_html)
    print(formatted_html)

    # Test case 2
    input_html = """
    <html>
    <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <pre class="code">print('Hello, world!')</pre>
        <p>Another paragraph.</p>
        <pre class="code"><code>for i in range(5):
            print(i)</code></pre>
        </body>
    </html>
    """
    code_blocks = html_util.extract_code_from_html_text(input_html)
    print(code_blocks)
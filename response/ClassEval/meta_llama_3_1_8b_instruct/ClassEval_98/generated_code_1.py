import xml.etree.ElementTree as ET

class XMLProcessor:
    """
    This is a class as XML files handler, including reading, writing, processing as well as finding elements in a XML file.
    """

    def __init__(self, file_name):
        """
        Initialize the XMLProcessor object with the given file name.
        :param file_name:string, the name of the XML file to be processed.
        """
        self.file_name = file_name
        self.root = None

    def read_xml(self):
        """
        Reads the XML file and returns the root element.
        :return: Element, the root element of the XML file.
        """
        try:
            tree = ET.parse(self.file_name)
            self.root = tree.getroot()
            return self.root
        except FileNotFoundError:
            print("The file does not exist.")
            return None
        except ET.ParseError:
            print("Failed to parse the XML file.")
            return None

    def write_xml(self, file_name):
        """
        Writes the XML data to the specified file.
        :param file_name: string, the name of the file to write the XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        try:
            tree = ET.ElementTree(self.root)
            tree.write(file_name)
            return True
        except Exception as e:
            print(f"Failed to write the XML file: {str(e)}")
            return False

    def process_xml_data(self, file_name):
        """
        Modifies the data in XML elements and writes the updated XML data to a new file.
        :param file_name: string, the name of the file to write the modified XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        try:
            # Example: modify the text of all 'item' elements
            for item in self.root.findall('.//item'):
                item.text = item.text.upper()
            tree = ET.ElementTree(self.root)
            tree.write(file_name)
            return True
        except Exception as e:
            print(f"Failed to process the XML data: {str(e)}")
            return False

    def find_element(self, element_name):
        """
        Finds the XML elements with the specified name.
        :param element_name: string, the name of the elements to find.
        :return: list, a list of found elements with the specified name.
        """
        try:
            return self.root.findall('.//' + element_name)
        except Exception as e:
            print(f"Failed to find the XML elements: {str(e)}")
            return []

if __name__ == "__main__":
    # Test case for read_xml method
    xml_processor = XMLProcessor('test.xml')
    root_element = xml_processor.read_xml()
    if root_element is not None:
        print(f"Root element: {root_element.tag}")
    else:
        print("Failed to read the XML file.")

    # Test case for write_xml method
    xml_processor = XMLProcessor('test.xml')
    root = xml_processor.read_xml()
    success = xml_processor.write_xml('output.xml')
    print(f"Write XML success: {success}")

    # Test case for process_xml_data method
    xml_processor = XMLProcessor('test.xml')
    root = xml_processor.read_xml()
    success = xml_processor.process_xml_data('processed.xml')
    print(f"Process XML data success: {success}")

    # Test case for find_element method
    xml_processor = XMLProcessor('test.xml')
    root = xml_processor.read_xml()
    items = xml_processor.find_element('item')
    for item in items:
        print(item.text)
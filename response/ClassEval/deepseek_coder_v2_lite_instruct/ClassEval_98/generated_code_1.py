import xml.etree.ElementTree as ET

class XMLProcessor:
    """
    This is a class as XML files handler, including reading, writing, processing as well as finding elements in a XML file.
    """

    def __init__(self, file_name):
        """
        Initialize the XMLProcessor object with the given file name.
        :param file_name: string, the name of the XML file to be processed.
        """
        self.file_name = file_name
        self.root = None

    def read_xml(self):
        """
        Reads the XML file and returns the root element.
        :return: Element, the root element of the XML file.
        """
        tree = ET.parse(self.file_name)
        self.root = tree.getroot()
        return self.root

    def write_xml(self, output_file_name):
        """
        Writes the XML data to the specified file.
        :param output_file_name: string, the name of the file to write the XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        tree = ET.ElementTree(self.root)
        try:
            tree.write(output_file_name)
            return True
        except Exception as e:
            print(f"Error writing XML file: {e}")
            return False

    def process_xml_data(self, output_file_name):
        """
        Modifies the data in XML elements and writes the updated XML data to a new file.
        :param output_file_name: string, the name of the file to write the modified XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        # Example: Change all item names to uppercase
        for elem in self.root.findall('.//item'):
            elem.text = elem.text.upper() if elem.text else elem.text

        return self.write_xml(output_file_name)

    def find_element(self, element_name):
        """
        Finds the XML elements with the specified name.
        :param element_name: string, the name of the elements to find.
        :return: list, a list of found elements with the specified name.
        """
        elements = self.root.findall(f'.//{element_name}')
        return elements

# Test cases
if __name__ == "__main__":
    xml_processor = XMLProcessor('test.xml')
    root = xml_processor.read_xml()
    
    # Test read_xml
    print("Read XML:")
    print(root)
    
    # Test write_xml
    print("Write XML:")
    success = xml_processor.write_xml('output.xml')
    print(success)
    
    # Test process_xml_data
    print("Process XML Data:")
    success = xml_processor.process_xml_data('processed.xml')
    print(success)
    
    # Test find_element
    print("Find Elements:")
    items = xml_processor.find_element('item')
    for item in items:
        print(item.text)
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
        try:
            self.tree = ET.parse(self.file_name)
            self.root = self.tree.getroot()
            return self.root
        except ET.ParseError as e:
            print(f"Error parsing XML file: {str(e)}")
            return None

    def write_xml(self, file_name):
        """
        Writes the XML data to the specified file.
        :param file_name: string, the name of the file to write the XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        try:
            self.tree.write(file_name)
            return True
        except ET.ParseError as e:
            print(f"Error writing XML file: {str(e)}")
            return False

    def process_xml_data(self, file_name):
        """
        Modifies the data in XML elements and writes the updated XML data to a new file.
        :param file_name: string, the name of the file to write the modified XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        modified_root = self.modify_elements(self.root)
        if modified_root:
            self.write_xml(file_name)
            return True
        return False

    def modify_elements(self, element):
        """
        Helper method to modify elements in the XML tree.
        :param element: Element, the element to modify.
        :return: Element or None, the modified element or None if not modified.
        """
        for child in element:
            new_value = input(f"Enter new value for {child.tag}: ")
            child.text = new_value
        return element

    def find_element(self, element_name):
        """
        Finds the XML elements with the specified name.
        :param element_name: string, the name of the elements to find.
        :return: list, a list of found elements with the specified name.
        """
        found_elements = []
        for element in self.root.findall(element_name):
            found_elements.append(element)
        return found_elements

# Test cases
if __name__ == "__main__":
    
    xml_processor = XMLProcessor('test.xml')
    root = xml_processor.read_xml()
    if root is not None:
        print("Root tag:", root.tag)
        success = xml_processor.write_xml('output.xml')
        print("Write to output.xml:", success)
        items = xml_processor.find_element('item')
        print(f"Found {len(items)} elements with tag 'item':")
        for item in items:
            print(item.text)
    else:
        print("Failed to read XML.")
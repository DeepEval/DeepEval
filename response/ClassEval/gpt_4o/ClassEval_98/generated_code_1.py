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
            tree = ET.parse(self.file_name)
            self.root = tree.getroot()
            return self.root
        except Exception as e:
            print(f"Error reading XML file: {e}")
            return None

    def write_xml(self, file_name):
        """
        Writes the XML data to the specified file.
        :param file_name: string, the name of the file to write the XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        if self.root is None:
            print("Root is not defined. Cannot write XML.")
            return False

        try:
            tree = ET.ElementTree(self.root)
            tree.write(file_name)
            return True
        except Exception as e:
            print(f"Error writing XML file: {e}")
            return False

    def process_xml_data(self, file_name):
        """
        Modifies the data in XML elements and writes the updated XML data to a new file.
        :param file_name: string, the name of the file to write the modified XML data.
        :return: bool, True if the write operation is successful, False otherwise.
        """
        if self.root is None:
            print("Root is not defined. Cannot process XML.")
            return False

        try:
            # Example modification: append '_processed' to each element's text
            for elem in self.root.iter():
                if elem.text:
                    elem.text = elem.text.strip() + "_processed"

            return self.write_xml(file_name)
        except Exception as e:
            print(f"Error processing XML data: {e}")
            return False

    def find_element(self, element_name):
        """
        Finds the XML elements with the specified name.
        :param element_name: string, the name of the elements to find.
        :return: list, a list of found elements with the specified name.
        """
        if self.root is None:
            print("Root is not defined. Cannot find elements.")
            return []

        return self.root.findall(element_name)


if __name__ == "__main__":
    # Test case for read_xml
    xml_processor = XMLProcessor('test.xml')
    root_element = xml_processor.read_xml()
    print(f"Root Element: {root_element}")

    # Test case for write_xml
    success_write = xml_processor.write_xml('output.xml')
    print(f"Write Success: {success_write}")

    # Test case for process_xml_data
    success_process = xml_processor.process_xml_data('processed.xml')
    print(f"Process Success: {success_process}")

    # Test case for find_element
    items = xml_processor.find_element('item')
    for item in items:
        print(f"Item Text: {item.text}")
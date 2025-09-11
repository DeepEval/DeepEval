import csv

class CSVProcessor:
    """
    This is a class for processing CSV files, including reading and writing CSV data,
    as well as processing specific operations and saving as a new CSV file.
    """

    def __init__(self):
        pass

    def read_csv(self, file_name):
        """
        Read the csv file by file_name, get the title and data from it.
        :param file_name: str, name of the csv file
        :return title, data: (list, list), first row is title, the rest is data
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.read_csv('read_test.csv')
        (['a', 'b', 'c', 'd'], [['hElLo', 'YoU', 'ME', 'LoW']])
        """
        with open(file_name, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            title = next(reader)  # Get the header
            data = [row for row in reader]  # Get the rest of the data
        return title, data

    def write_csv(self, data, file_name):
        """
        Write data into a csv file.
        :param data: list of lists, the data to write into the csv file
        :param file_name: str, name of the csv file
        :return:int, if success return 1, or 0 otherwise
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.write_csv([['a', 'b', 'c', 'd'], ['1', '2', '3', '4']], 'write_test.csv')
        1
        """
        try:
            with open(file_name, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
            return 1
        except:
            return 0

    def process_csv_data(self, N, save_file_name):
        """
        Read a csv file into variable title and data.
        Only remain the N th (from 0) column data and Capitalize them,
        store the title and new data into a new csv file.
        Add '_process' suffix after old file name, as a new file name.
        :param N: int, the N th column(from 0)
        :param save_file_name, the name of file that needs to be processed.
        :return:int, if success return 1, or 0 otherwise
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.read_csv('read_test.csv')
        (['a', 'b', 'c', 'd'], [['hElLo', 'YoU', 'ME', 'LoW']])
        >>> csvProcessor.process_csv_data(0, 'read_test.csv')
        1
        >>> csvProcessor.read_csv('read_test_process.csv')
        (['a', 'b', 'c', 'd'], [['HELLO']])
        """
        try:
            title, data = self.read_csv(save_file_name)
            new_data = [[row[N].upper()] for row in data]  # Capitalizing the Nth column
            new_file_name = f"{save_file_name.split('.')[0]}_process.csv"
            self.write_csv([title], new_file_name)  # Write the title first
            self.write_csv(new_data, new_file_name)  # Then write the new data
            return 1
        except:
            return 0


# Test cases
if __name__ == "__main__":
    # Create a CSVProcessor instance
    csvProcessor = CSVProcessor()

    # Test read_csv
    title, data = csvProcessor.read_csv('read_test.csv')
    print("Read CSV Output:", (title, data))

    # Test write_csv
    write_result = csvProcessor.write_csv([['a', 'b', 'c', 'd'], ['1', '2', '3', '4']], 'write_test.csv')
    print("Write CSV Output:", write_result)

    # Test process_csv_data
    process_result = csvProcessor.process_csv_data(0, 'read_test.csv')
    print("Process CSV Data Output:", process_result)

    # Verify the processed file
    processed_title, processed_data = csvProcessor.read_csv('read_test_process.csv')
    print("Processed CSV Output:", (processed_title, processed_data))
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
        Read the csv file by file_name, get the title and data from it
        :param file_name: str, name of the csv file
        :return title, data: (list, list), first row is title, the rest is data
        """
        with open(file_name, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            title = next(csv_reader)
            data = [row for row in csv_reader]
        return title, data

    def write_csv(self, data, file_name):
        """
        Write data into a csv file.
        :param file_name: str, name of the csv file
        :return: int, if success return 1, or 0 otherwise
        """
        try:
            with open(file_name, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(data)
            return 1
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0

    def process_csv_data(self, N, save_file_name):
        """
        Read a csv file into variable title and data.
        Only remain the Nth (from 0) column data and capitalize them, store the title and new data into a new csv file.
        Add '_process' suffix after the old file name, as a new file name.
        :param N: int, the Nth column(from 0)
        :param save_file_name: the name of the file that needs to be processed.
        :return: int, if success return 1, or 0 otherwise
        """
        try:
            title, data = self.read_csv(save_file_name)
            new_data = [[row[N].upper()] for row in data]
            new_title = [title[N]]

            new_file_name = save_file_name.rsplit('.', 1)
            new_file_name = f"{new_file_name[0]}_process.{new_file_name[1]}"

            return self.write_csv([new_title] + new_data, new_file_name)
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0

if __name__ == "__main__":
    # Instance of CSVProcessor
    csvProcessor = CSVProcessor()

    # Test read_csv
    output = csvProcessor.read_csv('read_test.csv')
    print(output)  # Expected: (['a', 'b', 'c', 'd'], [['hElLo', 'YoU', 'ME', 'LoW']])

    # Test write_csv
    output = csvProcessor.write_csv([['a', 'b', 'c', 'd'], ['1', '2', '3', '4']], 'write_test.csv')
    print(output)  # Expected: 1

    # Test process_csv_data
    output = csvProcessor.process_csv_data(0, 'read_test.csv')
    print(output)  # Expected: 1

    # Test reading processed CSV
    output = csvProcessor.read_csv('read_test_process.csv')
    print(output)  # Expected: (['a'], [['HELLO']])
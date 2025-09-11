import csv

class CSVProcessor:
    """
    This is a class for processing CSV files, including reading and writing CSV data, as well as processing specific operations and saving as a new CSV file.
    """

    def __init__(self):
        pass

    def read_csv(self, file_name):
        """
        Read the csv file by file_name, get the title and data from it
        :param file_name: str, name of the csv file
        :return title, data: (list, list), first row is title, the rest is data
        """
        title = []
        data = []
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if not title:
                    title = row
                else:
                    data.append(row)
        return title, data

    def write_csv(self, data, file_name):
        """
        Write data into a csv file.
        :param file_name: str, name of the csv file
        :return: int, if success return 1, or 0 otherwise
        """
        try:
            with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return 1
        except Exception as e:
            print(f"Error writing CSV: {e}")
            return 0

    def process_csv_data(self, N, save_file_name):
        """
        Read a csv file into variable title and data.
        Only remain the N th (from 0) column data and Capitalize them, store the title and new data into a new csv file.
        Add '_process' suffix after old file name, as a new file name.
        :param N: int, the N th column(from 0)
        :param save_file_name: str, the name of file that needs to be processed.
        :return: int, if success return 1, or 0 otherwise
        """
        title, data = self.read_csv(save_file_name)
        if N >= len(title):
            print("Column index out of range")
            return 0

        new_title = [title[N]]
        new_data = [[row[N].capitalize() for row in data]]

        new_file_name = save_file_name.rsplit('.', 1)[0] + '_process.csv'
        result = self.write_csv(new_data + [new_title], new_file_name)
        return result

# Test cases
if __name__ == "__main__":
    csvProcessor = CSVProcessor()
    
    # Test read_csv
    title, data = csvProcessor.read_csv('read_test.csv')
    print("Read CSV:", title, data)

    # Test write_csv
    result = csvProcessor.write_csv([['a', 'b', 'c', 'd'], ['1', '2', '3', '4']], 'write_test.csv')
    print("Write CSV:", result)

    # Test process_csv_data
    result = csvProcessor.process_csv_data(0, 'read_test.csv')
    print("Process CSV:", result)
    title, data = csvProcessor.read_csv('read_test_process.csv')
    print("Read Processed CSV:", title, data)
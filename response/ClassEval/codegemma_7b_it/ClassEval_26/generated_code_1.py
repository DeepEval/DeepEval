# mjt 修改

import csv

class CSVProcessor:
    """
    This is a class for processing CSV files, including readring and writing CSV data, as well as processing specific operations and saving as a new CSV file.
    """


    def __init__(self):
        pass

    def read_csv(self, file_name):
        """
        Read the csv file by file_name, get the title and data from it
        :param file_name: str, name of the csv file
        :return title, data: (list, list), first row is title, the rest is data
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.read_csv('read_test.csv')
        (['a', 'b', 'c', 'd'], [['hElLo', 'YoU', 'ME', 'LoW']])
        """
        title = []
        data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
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
        :return:int, if success return 1, or 0 otherwise
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.write_csv([['a', 'b', 'c', 'd'], ['1', '2', '3', '4']], 'write_test.csv')
        1
        """
        try:
            with open(file_name, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            return 1
        except Exception as e:
            print(e)
            return 0

    def process_csv_data(self, N, save_file_name):
        """
        Read a csv file into variable title and data.
        Only remain the N th (from 0) column data and Capitalize them, store the title and new data into a new csv file.
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
            new_data = []
            for row in data:
                new_row = [row[N].upper()]
                new_data.append(new_row)
            self.write_csv([title, *new_data], save_file_name + '_process')
            return 1
        except Exception as e:
            print(e)
            return 0

if __name__ == "__main__":
    # csvProcessor = CSVProcessor()
    # # Test case for read_csv
    # title, data = csvProcessor.read_csv('read_test.csv')
    # print(title, data)

    # # Test case for write_csv
    # data = [['a', 'b', 'c', 'd'], ['1', '2', '3', '4']]
    # file_name = 'write_test.csv'
    # result = csvProcessor.write_csv(data, file_name)
    # print(result)

    # # Test case for process_csv_data
    # N = 0
    # save_file_name = 'read_test.csv'
    # result = csvProcessor.process_csv_data(N, save_file_name)
    # print(result)
    
    import tempfile
    import os

    csvProcessor = CSVProcessor()

    virtual_csv_content = """name,age,city
                Alice,25,New York
                Bob,30,London
                Charlie,35,Paris
                """

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
        temp_file.write(virtual_csv_content)
        temp_file_path = temp_file.name
    title, data = csvProcessor.read_csv(temp_file_path)
    os.remove(temp_file_path)

    write_data = [
        ['id', 'product', 'price'],
        ['1', 'Laptop', '999'],
        ['2', 'Phone', '699'],
        ['3', 'Tablet', '299']
    ]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
        temp_write_path = temp_file.name
    result = csvProcessor.write_csv(write_data, temp_write_path)
    with open(temp_write_path, 'r', encoding='utf-8') as f:
        print(f.read())
    os.remove(temp_write_path)
    print("-" * 50)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as temp_file:
        temp_file.write("name,age,city\nAlice,25,NY\nBob,30,LD\nCharlie,35,PR")
        temp_process_path = temp_file.name
    result = csvProcessor.process_csv_data(N=1, save_file_name=temp_process_path)
    processed_file = temp_process_path + '_process'
    with open(temp_process_path, 'r', encoding='utf-8') as f:
        print(f.read())
    os.remove(temp_process_path)
    os.remove(processed_file)

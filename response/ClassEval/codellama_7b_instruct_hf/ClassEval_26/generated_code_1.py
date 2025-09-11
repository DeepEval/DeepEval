import csv

class CSVProcessor:
    def __init__(self):
        pass

    def read_csv(self, file_name):
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            title = next(reader)
            data = list(reader)
            return title, data

    def write_csv(self, data, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)
        return 1

    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        new_data = []
        for row in data:
            new_data.append(row[N].capitalize())
        with open(save_file_name + '_process.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(title)
            writer.writerows(new_data)
        return 1
    
if __name__ == "__main__":
    csv_processor = CSVProcessor()
    title, data = csv_processor.read_csv("read_test.csv")
    print(title)
    print(data)
    csv_processor.write_csv(data, "write_test.csv")
    csv_processor.process_csv_data(0, "write_test.csv")
    print("CSV processing completed.")
    
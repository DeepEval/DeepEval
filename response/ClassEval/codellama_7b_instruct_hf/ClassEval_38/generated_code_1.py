import openpyxl

class ExcelProcessor:
    def __init__(self):
        pass

    def read_excel(self, file_name):
        wb = openpyxl.load_workbook(filename=file_name)
        sheet = wb.active
        data = []
        for row in sheet.rows:
            data.append([cell.value for cell in row])
        return data

    def write_excel(self, data, file_name):
        wb = openpyxl.Workbook()
        sheet = wb.active
        for row in data:
            sheet.append(row)
        wb.save(filename=file_name)
        return 1

    def process_excel_data(self, N, save_file_name):
        data = self.read_excel(save_file_name)
        for row in data:
            row[N] = row[N].upper()
        return self.write_excel(data, save_file_name)

if __name__ == "__main__":
    processor = ExcelProcessor()
    new_data = [
        ('Name', 'Age', 'Country'),
        ('John', 25, 'USA'),
        ('Alice', 30, 'Canada'),
        ('Bob', 35, 'Australia'),
        ('Julia', 28, 'Germany')
    ]
    success, output_file = processor.process_excel_data(1, 'test_data.xlsx')
    print(success)
    print(output_file)
    

class PageUtil:
    """
    PageUtil class is a versatile utility for handling pagination and search functionalities in an efficient and convenient manner.
    """

    def __init__(self, data, page_size):
        """
        Initialize the PageUtil object with the given data and page size.
        :param data: list, the data to be paginated
        :param page_size: int, the number of items per page
        """
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (self.total_items + page_size - 1) // page_size

    def get_page(self, page_number):
        """
        Retrieve a specific page of data.
        :param page_number: int, the page number to fetch
        :return: list, the data on the specified page
        """
        if page_number < 1 or page_number > self.total_pages:
            raise ValueError("Invalid page number")
        
        start_index = (page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        return self.data[start_index:end_index]

    def get_page_info(self, page_number):
        """
        Retrieve information about a specific page.
        :param page_number: int, the page number to fetch information about
        :return: dict, containing page information such as current page number, total pages, etc.
        """
        if page_number < 1 or page_number > self.total_pages:
            raise ValueError("Invalid page number")
        
        return {
            "current_page": page_number,
            "per_page": self.page_size,
            "total_pages": self.total_pages,
            "total_items": self.total_items,
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": self.get_page(page_number)
        }

    def search(self, keyword):
        """
        Search for items in the data that contain the given keyword.
        :param keyword: str, the keyword to search for
        :return: dict, containing search information such as total results and matching items
        """
        matching_items = [item for item in self.data if str(item).find(str(keyword))!= -1]
        total_pages = (len(matching_items) + self.page_size - 1) // self.page_size
        return {
            "keyword": keyword,
            "total_results": len(matching_items),
            "total_pages": total_pages,
            "results": matching_items
        }


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    page_util = PageUtil(data, 2)

    print("Test Case 1: Get Page")
    try:
        output = page_util.get_page(3)
        print(output)  # Output: [7, 8]
    except ValueError as e:
        print(e)

    print("\nTest Case 2: Get Page Info")
    output = page_util.get_page_info(3)
    print(output)  # Output: {'current_page': 3, 'per_page': 2, 'total_pages': 5, 'total_items': 10, 'has_previous': True, 'has_next': True, 'data': [7, 8]}

    print("\nTest Case 3: Search")
    output = page_util.search(5)
    print(output)  # Output: {'keyword': '5', 'total_results': 1, 'total_pages': 1,'results': [5]}
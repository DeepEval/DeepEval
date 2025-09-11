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
            return []
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
            return {}
        current_page_data = self.get_page(page_number)
        return {
            "current_page": page_number,
            "per_page": self.page_size,
            "total_pages": self.total_pages,
            "total_items": self.total_items,
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": current_page_data
        }

    def search(self, keyword):
        """
        Search for items in the data that contain the given keyword.
        :param keyword: str, the keyword to search for
        :return: dict, containing search information such as total results and matching items
        """
        # Assuming all items are string representations for simplicity
        matching_items = [item for item in self.data if keyword in str(item)]
        total_results = len(matching_items)
        total_pages = (total_results + self.page_size - 1) // self.page_size
        return {
            "keyword": keyword,
            "total_results": total_results,
            "total_pages": total_pages,
            "results": matching_items
        }

# Test cases
if __name__ == "__main__":
    page_util = PageUtil([1, 2, 3, 4], 1)

    # Test get_page method
    output = page_util.get_page(1)
    print("get_page(1):", output)  # Expected: [1]

    # Test get_page_info method
    page_info_output = page_util.get_page_info(1)
    print("get_page_info(1):", page_info_output)
    # Expected: {
    #     "current_page": 1,
    #     "per_page": 1,
    #     "total_pages": 4,
    #     "total_items": 4,
    #     "has_previous": False,
    #     "has_next": True,
    #     "data": [1]
    # }

    # Test search method
    search_output = page_util.search("1")
    print("search('1'):", search_output)
    # Expected: {
    #     "keyword": "1",
    #     "total_results": 1,
    #     "total_pages": 1,
    #     "results": [1]
    # }
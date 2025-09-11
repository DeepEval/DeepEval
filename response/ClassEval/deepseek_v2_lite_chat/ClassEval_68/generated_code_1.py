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
        start = (page_number - 1) * self.page_size
        end = start + self.page_size
        return self.data[start:end]

    def get_page_info(self, page_number):
        """
        Retrieve information about a specific page.
        :param page_number: int, the page number to fetch information about
        :return: dict, containing page information such as current page number, total pages, etc.
        """
        has_previous = page_number > 1
        has_next = page_number < self.total_pages
        current_page = page_number
        per_page = self.page_size
        total_pages = self.total_pages
        total_items = self.total_items
        results = self.get_page(page_number)
        return {
            "current_page": current_page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "has_previous": has_previous,
            "has_next": has_next,
            "data": results
        }

    def search(self, keyword):
        """
        Search for items in the data that contain the given keyword.
        :param keyword: str, the keyword to search for
        :return: dict, containing search information such as total results and matching items
        """
        results = [item for item in self.data if keyword.lower() in str(item)]
        return {
            "keyword": keyword,
            "total_results": len(results),
            "total_pages": 1,
            "results": results
        }

# Test cases
if __name__ == "__main__":
    page_util = PageUtil([1, 2, 3, 4], 1)
    print(page_util.get_page(1))  # [1]
    print(page_util.get_page_info(1))  # {...}
    print(page_util.search("1"))  # {...}
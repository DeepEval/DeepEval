import unittest

class PageUtil:
    def __init__(self, data, page_size):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (self.total_items + page_size - 1) // page_size

    def get_page(self, page_number):
        start_index = (page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        return self.data[start_index:end_index]

    def get_page_info(self, page_number):
        current_page = page_number
        total_pages = self.total_pages
        total_items = self.total_items
        has_previous = current_page > 1
        has_next = current_page < total_pages
        data = self.get_page(page_number)
        return {
            "current_page": current_page,
            "per_page": self.page_size,
            "total_pages": total_pages,
            "total_items": total_items,
            "has_previous": has_previous,
            "has_next": has_next,
            "data": data
        }

    def search(self, keyword):
        results = []
        for item in self.data:
            if keyword in str(item):
                results.append(item)
        total_results = len(results)
        total_pages = (total_results + self.page_size - 1) // self.page_size
        return {
            "keyword": keyword,
            "total_results": total_results,
            "total_pages": total_pages,
            "results": results
        }

class PageUtilTest(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.page_size = 2

    def test_get_page(self):
        page_util = PageUtil(self.data, self.page_size)
        self.assertEqual(page_util.get_page(1), [1, 2])
        self.assertEqual(page_util.get_page(2), [3, 4])
        self.assertEqual(page_util.get_page(3), [5, 6])
        self.assertEqual(page_util.get_page(4), [7, 8])
        self.assertEqual(page_util.get_page(5), [9, 10])

    def test_get_page_info(self):
        page_util = PageUtil(self.data, self.page_size)
        self.assertEqual(page_util.get_page_info(1), {
            "current_page": 1,
            "per_page": self.page_size,
            "total_pages": 5,
            "total_items": 10,
            "has_previous": False,
            "has_next": True,
            "data": [1, 2]
        })
        self.assertEqual(page_util.get_page_info(2), {
            "current_page": 2,
            "per_page": self.page_size,
            "total_pages": 5,
            "total_items": 10,
            "has_previous": True,
            "has_next": True,
            "data": [3, 4]
        })
        self.assertEqual(page_util.get_page_info(3), {
            "current_page": 3,
            "per_page": self.page_size,
            "total_pages": 5,
            "total_items": 10,
            "has_previous": True,
            "has_next": True,
            "data": [5, 6]
        })
        self.assertEqual(page_util.get_page_info(4), {
            "current_page": 4,
            "per_page": self.page_size,
            "total_pages": 5,
            "total_items": 10,
            "has_previous": True,
            "has_next": True,
            "data": [7, 8]
        })
        self.assertEqual(page_util.get_page_info(5), {
            "current_page": 5,
            "per_page": self.page_size,
            "total_pages": 5,
            "total_items": 10,
            "has_previous": True,
            "has_next": False,
            "data": [9, 10]
        })

    def test_search(self):
        page_util = PageUtil(self.data, self.page_size)
        self.assertEqual(page_util.search("1"), {
            "keyword": "1",
            "total_results": 1,
            "total_pages": 1,
            "results": [1]
        })
        self.assertEqual(page_util.search("2"), {
            "keyword": "2",
            "total_results": 1,
            "total_pages": 1,
            "results": [2]
        })
        self.assertEqual(page_util.search("3"), {
            "keyword": "3",
            "total_results": 1,
            "total_pages": 1,
            "results": [3]
        })
        self.assertEqual(page_util.search("4"), {
            "keyword": "4",
            "total_results": 1,
            "total_pages": 1,
            "results": [4]
        })
        self.assertEqual(page_util.search("5"), {
            "keyword": "5",
            "total_results": 1,
            "total_pages": 1,
            "results": [5]
        })
        self.assertEqual(page_util.search("6"), {
            "keyword": "6",
            "total_results": 1,
            "total_pages": 1,
            "results": [6]
        })
        self.assertEqual(page_util.search("7"), {
            "keyword": "7",
            "total_results": 1,
            "total_pages": 1,
            "results": [7]
        })
        self.assertEqual(page_util.search("8"), {
            "keyword": "8",
            "total_results": 1,
            "total_pages": 1,
            "results": [8]
        })
        self.assertEqual(page_util.search("9"), {
            "keyword": "9",
            "total_results": 1,
            "total_pages": 1,
            "results": [9]
        })
        self.assertEqual(page_util.search("10"), {
            "keyword": "10",
            "total_results": 1,
            "total_pages": 1,
            "results": [10]
        })

if __name__ == "__main__":
    unittest.main()
import datetime
import time

class TimeUtils:
    """
    This is a time util class, including getting the current time and date, adding seconds to a datetime, converting between strings and datetime objects, calculating the time difference in minutes, and formatting a datetime object.
    """

    def __init__(self):
        """
        Get the current datetime
        """
        self.datetime = datetime.datetime.now()

    def get_current_time(self):
        """
        Return the current time in the format of '%H:%M:%S'
        :return: string
        """
        return self.datetime.strftime('%H:%M:%S')

    def get_current_date(self):
        """
        Return the current date in the format of "%Y-%m-%d"
        :return: string
        """
        return self.datetime.strftime('%Y-%m-%d')

    def add_seconds(self, seconds):
        """
        Add the specified number of seconds to the current time
        :param seconds: int, number of seconds to add
        :return: string, time after adding the specified number of seconds in the format '%H:%M:%S'
        """
        new_datetime = self.datetime + datetime.timedelta(seconds=seconds)
        return new_datetime.strftime('%H:%M:%S')

    def string_to_datetime(self, string):
        """
        Convert the time string to a datetime instance
        :param string: string, string before converting format
        :return: datetime instance
        """
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

    def datetime_to_string(self, datetime):
        """
        Convert a datetime instance to a string
        :param datetime: the datetime instance to convert
        :return: string, converted time string
        """
        return datetime.strftime('%Y-%m-%d %H:%M:%S')

    def get_minutes(self, string_time1, string_time2):
        """
        Calculate how many minutes have passed between two times, and round the results to the nearest
        :return: int, the number of minutes between two times, rounded off
        """
        datetime1 = self.string_to_datetime(string_time1)
        datetime2 = self.string_to_datetime(string_time2)
        minutes = abs((datetime2 - datetime1).total_seconds() / 60)
        return round(minutes)

    def get_format_time(self, year, month, day, hour, minute, second):
        """
        get format time
        :param year: int
        :param month: int
        :param day: int
        :param hour: int
        :param minute: int
        :param second: int
        :return: formatted time string
        """
        return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

if __name__ == "__main__":
    timeutils = TimeUtils()

    # Test case for get_current_time
    print("Test case for get_current_time:")
    print(f"Output: {timeutils.get_current_time()}")

    # Test case for get_current_date
    print("\nTest case for get_current_date:")
    print(f"Output: {timeutils.get_current_date()}")

    # Test case for add_seconds
    print("\nTest case for add_seconds:")
    print(f"Output: {timeutils.add_seconds(600)}")

    # Test case for string_to_datetime
    print("\nTest case for string_to_datetime:")
    print(f"Output: {timeutils.string_to_datetime('2023-06-14 19:30:03')}")

    # Test case for datetime_to_string
    print("\nTest case for datetime_to_string:")
    print(f"Output: {timeutils.datetime_to_string(timeutils.datetime)}")

    # Test case for get_minutes
    print("\nTest case for get_minutes:")
    print(f"Output: {timeutils.get_minutes('2001-07-18 01:01:01', '2001-07-18 02:01:01')}")

    # Test case for get_format_time
    print("\nTest case for get_format_time:")
    print(f"Output: {timeutils.get_format_time(2001, 7, 18, 1, 1, 1)}")
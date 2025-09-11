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
        new_time = self.datetime + datetime.timedelta(seconds=seconds)
        return new_time.strftime('%H:%M:%S')

    def string_to_datetime(self, string):
        """
        Convert the time string to a datetime instance
        :param string: string, string before converting format
        :return: datetime instance
        """
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

    def datetime_to_string(self, datetime_obj):
        """
        Convert a datetime instance to a string
        :param datetime_obj: the datetime instance to convert
        :return: string, converted time string
        """
        return datetime_obj.strftime('%Y-%m-%d %H:%M:%S')

    def get_minutes(self, string_time1, string_time2):
        """
        Calculate how many minutes have passed between two times, and round the results to the nearest
        :return: int, the number of minutes between two times, rounded off
        """
        dt1 = self.string_to_datetime(string_time1)
        dt2 = self.string_to_datetime(string_time2)
        delta = dt2 - dt1
        return round(delta.total_seconds() / 60)

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
        dt = datetime.datetime(year, month, day, hour, minute, second)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    timeutils = TimeUtils()

    # Test case for get_current_time
    print("Current Time:", timeutils.get_current_time())  # Example output: "Current Time: 19:19:22"

    # Test case for get_current_date
    print("Current Date:", timeutils.get_current_date())  # Example output: "Current Date: 2023-06-14"

    # Test case for add_seconds
    print("Time after adding seconds:", timeutils.add_seconds(600))  # Example output: "19:29:22"

    # Test case for string_to_datetime
    dt = timeutils.string_to_datetime("2001-7-18 1:1:1")
    print("Datetime:", dt)  # Example output: "Datetime: 2001-07-18 01:01:01"

    # Test case for datetime_to_string
    print("Datetime to String:", timeutils.datetime_to_string(dt))  # Example output: "2001-07-18 01:01:01"

    # Test case for get_minutes
    minutes = timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 2:1:1")
    print("Minutes Difference:", minutes)  # Example output: "Minutes Difference: 60"

    # Test case for get_format_time
    formatted_time = timeutils.get_format_time(2001, 7, 18, 1, 1, 1)
    print("Formatted Time:", formatted_time)  # Example output: "Formatted Time: 2001-07-18 01:01:01"
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
        time1 = self.string_to_datetime(string_time1)
        time2 = self.string_to_datetime(string_time2)
        time_diff = int((time2 - time1).total_seconds() / 60)
        return round(time_diff)

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
        new_datetime = datetime.datetime(year, month, day, hour, minute, second)
        return new_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
if __name__ == "__main__":
    time_util = TimeUtils()
    print("Current Time:", time_util.get_current_time())
    print("Current Date:", time_util.get_current_date())
    print("Time after adding 5000 seconds:", time_util.add_seconds(5000))
    time_str1 = "2023-07-31 10:00:00"
    time_str2 = "2023-07-31 12:30:00"
    print("Minutes between two times:", time_util.get_minutes(time_str1, time_str2))
    formatted_time = time_util.get_format_time(2023, 7, 31, 15, 45, 30)
    print("Formatted Time:", formatted_time)
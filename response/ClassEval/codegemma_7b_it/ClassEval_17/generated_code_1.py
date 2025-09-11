from datetime import datetime, timedelta

class CalendarUtil:
    """
    This is a class as CalendarUtil that provides functionalities to manage calendar
    events, schedule appointments, and perform conflict checks.
    """

    def __init__(self):
        """
        Initialize the calendar with an empty list of events.
        """
        self.events = []

    def add_event(self, event):
        """
        Add an event to the calendar.
        :param event: The event to be added to the calendar,dict.
        >>> calendar = CalendarUtil()
        >>> calendar.add_event({"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
        "New Year"})
        >>> calendar.events
        [{"date": datetime.datetime(2023, 1, 1, 0, 0), "start_time": datetime.datetime(2023,
        1, 1, 0, 0), "end_time": datetime.datetime(2023, 1, 1, 1, 0), "description":
        "New Year"}]

        """
        self.events.append(event)

    def remove_event(self, event):
        """
        Remove an event from the calendar.
        :param event: The event to be removed from the calendar,dict.
        >>> calendar = CalendarUtil()
        >>> calendar.events = [{"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
        "New Year"}]
        >>> calendar.remove_event({"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
        "New Year"})
        >>> calendar.events
        []

        """
        self.events.remove(event)

    def get_events(self, date):
        """
        Get all events on a given date.
        :param date: The date to get events for,datetime.
        :return: A list of events on the given date,list.
        >>> calendar = CalendarUtil()
        >>> calendar.events = [{"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
        "New Year"}]
        >>> calendar.get_events(datetime(2023, 1, 1, 0, 0))
        [{"date": datetime.datetime(2023, 1, 1, 0, 0), "start_time": datetime.datetime(2023,
        1, 1, 0, 0), "end_time": datetime.datetime(2023, 1, 1, 1, 0), "description":
        "New Year"}]

        """
        return [event for event in self.events if event["date"] == date]

    def is_available(self, start_time, end_time):
        """
        Check if the calendar is available for a given time slot.
        :param start_time: The start time of the time slot,datetime.
        :param end_time: The end time of the time slot,datetime.
        :return: True if the calendar is available for the given time slot, False
        otherwise,bool.
        >>> calendar = CalendarUtil()
        >>> calendar.events = [{"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
        "New Year"}]
        >>> calendar.is_available(datetime(2023, 1, 1, 0, 0), datetime(2023, 1,
        1, 1, 0))
        False

        """
        for event in self.events:
            if event["start_time"] <= start_time and event["end_time"] >= end_time:
                return False
        return True

    def get_available_slots(self, date):
        """
        Get all available time slots on a given date.
        :param date: The date to get available time slots for,datetime.
        :return: A list of available time slots on the given date,list.
        >>> calendar = CalendarUtil()
        >>> calendar.events = [{"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 23, 0), "description":
        "New Year"}]
        >>> calendar.get_available_slots(datetime(2023, 1, 1))
        [(datetime.datetime(2023, 1, 1, 23, 0), datetime.datetime(2023, 1, 2, 0,
        0))]

        """
        events = self.get_events(date)
        if not events:
            return [(datetime.combine(date, datetime.min), datetime.combine(date,
        datetime.max))]
        events.sort(key=lambda event: event["start_time"])
        available_slots = []
        for i in range(len(events)):
            if i == 0:
                available_slots.append((datetime.combine(date, datetime.min), events[i]["start_time"]))
            if i == len(events) - 1:
                available_slots.append((events[i]["end_time"], datetime.combine(date,
        datetime.max)))
            else:
                available_slots.append((events[i]["end_time"], events[i + 1]["start_time"]))
        return available_slots

    def get_upcoming_events(self, num_events):
        """
        Get the next n upcoming events from a given date.
        :param date: The date to get upcoming events from,datetime.
        :param n: The number of upcoming events to get,int.
        :return: A list of the next n upcoming events from the given date,list.
        >>> calendar = CalendarUtil()
        >>> calendar.events = [{"date": datetime(2023, 1, 1, 0, 0), "start_time":
        datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 23, 0), "description":
        "New Year"},{"date": datetime(2023, 1, 2, 0, 0),"end_time": datetime(2023,
        1, 2, 1, 0), "description": "New Year 2"}]
        >>> calendar.get_upcoming_events(1)
        [{"date": datetime.datetime(2023, 1, 1, 0, 0), "start_time": datetime.datetime(2023,
        1, 1, 0, 0), "end_time": datetime.datetime(2023, 1, 1, 23, 0), "description":
        "New Year"}, {"date": datetime.datetime(2023, 1, 2, 0, 0), "end_time": datetime.datetime(2023,
        1, 2, 1, 0), "description": "New Year 2"}]

        """
        return sorted([event for event in self.events if event["date"] >= date],
        key=lambda event: event["date"])[:num_events]

if __name__ == "__main__":
    calendar = CalendarUtil()
    calendar.add_event({"date": datetime(2023, 1, 1, 0, 0), "start_time":
    datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
    "New Year"})
    calendar.get_events(datetime(2023, 1, 1, 0, 0))
    calendar.is_available(datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0))
    calendar.get_available_slots(datetime(2023, 1, 1))
    calendar.remove_event({"date": datetime(2023, 1, 1, 0, 0), "start_time":
    datetime(2023, 1, 1, 0, 0), "end_time": datetime(2023, 1, 1, 1, 0), "description":
    "New Year"})
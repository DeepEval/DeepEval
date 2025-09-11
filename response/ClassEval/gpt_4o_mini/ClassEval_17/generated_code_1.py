from datetime import datetime, timedelta

class CalendarUtil:
    """
    This is a class as CalendarUtil that provides functionalities to manage calendar events, schedule appointments, and perform conflict checks.
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
        """
        self.events.append(event)

    def remove_event(self, event):
        """
        Remove an event from the calendar.
        :param event: The event to be removed from the calendar,dict.
        """
        if event in self.events:
            self.events.remove(event)

    def get_events(self, date):
        """
        Get all events on a given date.
        :param date: The date to get events for,datetime.
        :return: A list of events on the given date,list.
        """
        return [event for event in self.events if event['date'] == date]

    def is_available(self, start_time, end_time):
        """
        Check if the calendar is available for a given time slot.
        :param start_time: The start time of the time slot,datetime.
        :param end_time: The end time of the time slot,datetime.
        :return: True if the calendar is available for the given time slot, False otherwise,bool.
        """
        for event in self.events:
            if event['date'] == start_time.date():
                if not (end_time <= event['start_time'] or start_time >= event['end_time']):
                    return False
        return True

    def get_available_slots(self, date):
        """
        Get all available time slots on a given date.
        :param date: The date to get available time slots for,datetime.
        :return: A list of available time slots on the given date,list.
        """
        slots = []
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())
        
        busy_times = [(event['start_time'], event['end_time']) for event in self.events if event['date'] == date]
        
        if not busy_times:
            return [(start_of_day, end_of_day)]

        busy_times.sort()
        last_end_time = start_of_day
        
        for start, end in busy_times:
            if last_end_time < start:
                slots.append((last_end_time, start))
            last_end_time = max(last_end_time, end)

        if last_end_time < end_of_day:
            slots.append((last_end_time, end_of_day))

        return slots

    def get_upcoming_events(self, num_events):
        """
        Get the next n upcoming events from a given date.
        :param num_events: The number of upcoming events to get,int.
        :return: A list of the next n upcoming events from the current date,list.
        """
        now = datetime.now()
        upcoming_events = [event for event in self.events if event['start_time'] >= now]
        return upcoming_events[:num_events]

# Testing the CalendarUtil class
if __name__ == "__main__":
    calendar = CalendarUtil()

    # Test add_event
    calendar.add_event({'date': datetime(2023, 1, 1, 0, 0), 'start_time': datetime(2023, 1, 1, 0, 0), 'end_time': datetime(2023, 1, 1, 1, 0), 'description': 'New Year'})
    print(calendar.events)

    # Test remove_event
    calendar.remove_event({'date': datetime(2023, 1, 1, 0, 0), 'start_time': datetime(2023, 1, 1, 0, 0), 'end_time': datetime(2023, 1, 1, 1, 0), 'description': 'New Year'})
    print(calendar.events)

    # Test get_events
    calendar.add_event({'date': datetime(2023, 1, 1, 0, 0), 'start_time': datetime(2023, 1, 1, 0, 0), 'end_time': datetime(2023, 1, 1, 1, 0), 'description': 'New Year'})
    events_on_date = calendar.get_events(datetime(2023, 1, 1, 0, 0))
    print(events_on_date)

    # Test is_available
    availability = calendar.is_available(datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0))
    print(availability)

    # Test get_available_slots
    available_slots = calendar.get_available_slots(datetime(2023, 1, 1))
    print(available_slots)

    # Test get_upcoming_events
    calendar.add_event({'date': datetime(2023, 1, 2, 0, 0), 'start_time': datetime(2023, 1, 2, 0, 0), 'end_time': datetime(2023, 1, 2, 1, 0), 'description': 'New Year 2'})
    upcoming_events = calendar.get_upcoming_events(1)
    print(upcoming_events)
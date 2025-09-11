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
        :param event: The event to be added to the calendar, dict.
        """
        self.events.append(event)

    def remove_event(self, event):
        """
        Remove an event from the calendar.
        :param event: The event to be removed from the calendar, dict.
        """
        self.events = [e for e in self.events if e != event]

    def get_events(self, date):
        """
        Get all events on a given date.
        :param date: The date to get events for, datetime.
        :return: A list of events on the given date, list.
        """
        return [e for e in self.events if e['date'] == date]

    def is_available(self, start_time, end_time):
        """
        Check if the calendar is available for a given time slot.
        :param start_time: The start time of the time slot, datetime.
        :param end_time: The end time of the time slot, datetime.
        :return: True if the calendar is available for the given time slot, False otherwise, bool.
        """
        for event in self.events:
            if event['date'] == start_time.date():
                if event['start_time'] < end_time and event['end_time'] > start_time:
                    return False
        return True

    def get_available_slots(self, date):
        """
        Get all available time slots on a given date.
        :param date: The date to get available time slots for, datetime.
        :return: A list of available time slots on the given date, list.
        """
        available_slots = []
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = start_of_day + timedelta(days=1)

        current_time = start_of_day
        while current_time < end_of_day:
            if self.is_available(current_time, current_time + timedelta(hours=1)):
                next_time = current_time + timedelta(hours=1)
                available_slots.append((current_time, next_time))
            current_time += timedelta(minutes=15)  # Increment by 15 minutes

        return available_slots

    def get_upcoming_events(self, num_events):
        """
        Get the next n upcoming events from a given date.
        :param date: The date to get upcoming events from, datetime.
        :param num_events: The number of upcoming events to get, int.
        :return: A list of the next n upcoming events from the given date, list.
        """
        today = datetime.now().date()
        upcoming_events = [e for e in self.events if e['date'].date() >= today][:num_events]
        return upcoming_events

if __name__ == "__main__":
    calendar = CalendarUtil()
    calendar.add_event({'date': datetime(2023, 1, 1, 0, 0), 'start_time': datetime(2023, 1, 1, 0, 0), 'end_time': datetime(2023, 1, 1, 1, 0), 'description': 'New Year'})
    print(calendar.events)  # Test add_event
    calendar.remove_event({'date': datetime(2023, 1, 1, 0, 0), 'start_time': datetime(2023, 1, 1, 0, 0), 'end_time': datetime(2023, 1, 1, 1, 0), 'description': 'New Year'})
    print(calendar.events)  # Test remove_event
    events_on_date = calendar.get_events(datetime(2023, 1, 1, 0, 0))
    print(events_on_date)  # Test get_events
    print(calendar.is_available(datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0)))  # Test is_available
    available_slots = calendar.get_available_slots(datetime(2023, 1, 1))
    print(available_slots)  # Test get_available_slots
    upcoming_events = calendar.get_upcoming_events(1)
    print(upcoming_events)  # Test get_upcoming_events
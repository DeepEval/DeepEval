import datetime

class CalendarUtil:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")
        self.events.append(event)

    def remove_event(self, event):
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")
        self.events.remove(event)

    def get_events(self, date):
        if not isinstance(date, datetime.datetime):
            raise ValueError("date must be a datetime object")
        return [event for event in self.events if event["date"] == date]

    def is_available(self, start_time, end_time):
        if not isinstance(start_time, datetime.datetime):
            raise ValueError("start_time must be a datetime object")
        if not isinstance(end_time, datetime.datetime):
            raise ValueError("end_time must be a datetime object")
        for event in self.events:
            if event["start_time"] <= start_time < event["end_time"] or event["start_time"] < end_time <= event["end_time"]:
                return False
        return True

    def get_available_slots(self, date):
        if not isinstance(date, datetime.datetime):
            raise ValueError("date must be a datetime object")
        slots = []
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        while start_time < end_time:
            if self.is_available(start_time, start_time + timedelta(hours=1)):
                slots.append((start_time, start_time + timedelta(hours=1)))
            start_time += timedelta(hours=1)
        return slots

    def get_upcoming_events(self, num_events):
        if not isinstance(num_events, int):
            raise ValueError("num_events must be an integer")
        return self.events[:num_events]

if __name__ == "__main__":
    calendar = CalendarUtil()

    # Test add_event
    calendar.add_event({"date": datetime.datetime(2023, 1, 1), "start_time": datetime.datetime(2023, 1, 1, 0, 0), "end_time": datetime.datetime(2023, 1, 1, 1, 0), "description": "New Year"})
    output = calendar.get_events(datetime.datetime(2023, 1, 1))
    print(output)

    # Test remove_event
    calendar.remove_event({"date": datetime.datetime(2023, 1, 1), "start_time": datetime.datetime(2023, 1, 1, 0, 0), "end_time": datetime.datetime(2023, 1, 1, 1, 0), "description": "New Year"})
    output = calendar.get_events(datetime.datetime(2023, 1, 1))
    print(output)

    # Test get_events
    calendar.add_event({"date": datetime.datetime(2023, 1, 1), "start_time": datetime.datetime(2023, 1, 1, 0, 0), "end_time": datetime.datetime(2023, 1, 1, 1, 0), "description": "New Year"})
    calendar.add_event({"date": datetime.datetime(2023, 1, 2), "start_time": datetime.datetime(2023, 1, 2, 0, 0), "end_time": datetime.datetime(2023, 1, 2, 1, 0), "description": "New Year 2"})
    output = calendar.get_events(datetime.datetime(2023, 1, 1))
    print(output)

    # Test is_available
    output = calendar.is_available(datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 1, 0))
    print(output)

    # Test get_available_slots
    output = calendar.get_available_slots(datetime.datetime(2023, 1, 1))
    print(output)

    # Test get_upcoming_events
    output = calendar.get_upcoming_events(1)
    print(output)
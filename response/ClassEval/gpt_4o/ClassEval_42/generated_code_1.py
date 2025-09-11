class Hotel:
    """
    This is a class as hotel management system, managing the booking, check-in, check-out, and availability of rooms in a hotel with different room types.
    """

    def __init__(self, name, rooms):
        """
        Initialize the three fields in Hotel System.
        """
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def book_room(self, room_type, room_number, name):
        """
        Book rooms if available.
        """
        if room_type not in self.available_rooms or self.available_rooms[room_type] == 0:
            return False

        if room_number > self.available_rooms[room_type]:
            return self.available_rooms[room_type]

        self.available_rooms[room_type] -= room_number
        if room_type not in self.booked_rooms:
            self.booked_rooms[room_type] = {}
        if name not in self.booked_rooms[room_type]:
            self.booked_rooms[room_type][name] = 0
        self.booked_rooms[room_type][name] += room_number
        return 'Success!'

    def check_in(self, room_type, room_number, name):
        """
        Check in a guest.
        """
        if room_type not in self.booked_rooms or name not in self.booked_rooms[room_type]:
            return False

        if room_number > self.booked_rooms[room_type][name]:
            return False

        self.booked_rooms[room_type][name] -= room_number
        if self.booked_rooms[room_type][name] == 0:
            del self.booked_rooms[room_type][name]

    def check_out(self, room_type, room_number):
        """
        Check out rooms, add number for specific type in available_rooms.
        """
        if room_type not in self.available_rooms:
            self.available_rooms[room_type] = 0

        self.available_rooms[room_type] += room_number

    def get_available_rooms(self, room_type):
        """
        Get the number of specific type of available rooms.
        """
        return self.available_rooms.get(room_type, 0)


if __name__ == "__main__":
    # Test case for book_room
    hotel = Hotel('peace hotel', {'single': 5, 'double': 3})
    print(hotel.book_room('single', 1, 'guest 1'))  # Expected: 'Success!'
    print(hotel.book_room('single', 5, 'guest 1'))  # Expected: 4
    print(hotel.book_room('single', 4, 'guest 1'))  # Expected: 'Success!'
    print(hotel.book_room('single', 1, 'guest 1'))  # Expected: False
    print(hotel.book_room('triple', 1, 'guest 1'))  # Expected: False
    
    # Test case for check_in
    hotel = Hotel('peace hotel', {'single': 5, 'double': 3})
    hotel.book_room('single', 1, 'guest 1')
    print(hotel.check_in('single', 2, 'guest 1'))  # Expected: False
    hotel.check_in('single', 1, 'guest 1')
    print(hotel.booked_rooms)  # Expected: {'single': {}}

    # Test case for check_out
    hotel = Hotel('peace hotel', {'single': 5, 'double': 3})
    hotel.check_out('single', 2)
    print(hotel.available_rooms)  # Expected: {'single': 7, 'double': 3}
    hotel.check_out('triple', 2)
    print(hotel.available_rooms)  # Expected: {'single': 7, 'double': 3, 'triple': 2}

    # Test case for get_available_rooms
    hotel = Hotel('peace hotel', {'single': 5, 'double': 3})
    print(hotel.get_available_rooms('single'))  # Expected: 5
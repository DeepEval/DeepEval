class Hotel:
    """
    This is a class as hotel management system, managing the booking, check-in, check-out, and availability of rooms in a hotel with different room types.
    """

    def __init__(self, name, rooms):
        """
        Initialize the three fields in Hotel System.
        name is the hotel name.
        available_rooms stores the remaining rooms in the hotel
        booked_rooms stores the rooms that have been booked and the person's name who booked rooms.
        """
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def book_room(self, room_type, room_number, name):
        """
        Check if there are any rooms of the specified type available.
        if rooms are adequate, modify available_rooms and booked_rooms and finish booking, or fail to book otherwise.
        :param room_type: str
        :param room_number: int, the expected number of specified type rooms to be booked
        :param name: str, guest name
        :return: if number of rooms about to be booked doesn't exceed the remaining rooms, return str 'Success!'
                if exceeds but quantity of available rooms is not equal to zero, return int(the remaining quantity of this room type).
                if exceeds and quantity is zero or the room_type isn't in available_room, return False.
        """
        if room_type not in self.available_rooms:
            return False
        if room_number <= self.available_rooms[room_type]:
            self.available_rooms[room_type] -= room_number
            if room_type not in self.booked_rooms:
                self.booked_rooms[room_type] = {}
            if name in self.booked_rooms[room_type]:
                self.booked_rooms[room_type][name] += room_number
            else:
                self.booked_rooms[room_type][name] = room_number
            return 'Success!'
        else:
            remaining_rooms = self.available_rooms[room_type]
            return remaining_rooms if remaining_rooms > 0 else False

    def check_in(self, room_type, room_number, name):
        """
        Check if the room of the specified type and number is booked by the person named name.
        Remove this name when check in successfuly(room_number is equal to specific person's booked_rooms.
        """
        if room_type not in self.booked_rooms or name not in self.booked_rooms[room_type]:
            return False
        if room_number > self.booked_rooms[room_type][name]:
            return False
        self.booked_rooms[room_type][name] -= room_number
        if self.booked_rooms[room_type][name] == 0:
            del self.booked_rooms[room_type][name]
        return True

    def check_out(self, room_type, room_number):
        """
        Check out rooms, add number for specific type in available_rooms.
        If room_type is new, add new type in available_rooms.
        """
        if room_type not in self.available_rooms:
            self.available_rooms[room_type] = 0
        self.available_rooms[room_type] += room_number

    def get_available_rooms(self, room_type):
        """
        Get the number of specific type of available rooms.
        :param room_type: str, the room type that want to know
        :return: int, the remaining number of this type rooms.
        """
        return self.available_rooms.get(room_type, 0)

# Test cases
if __name__ == "__main__":
    hotel = Hotel('peace hotel', {'single': 5, 'double': 3})
    
    # Test book_room
    print(hotel.book_room('single', 1, 'guest 1'))  # Output: 'Success!'
    print(hotel.book_room('single', 5, 'guest 1'))  # Output: 4
    print(hotel.book_room('single', 4, 'guest 1'))  # Output: 'Success!'
    print(hotel.book_room('single', 1, 'guest 1'))  # Output: False
    print(hotel.book_room('triple', 1, 'guest 1'))  # Output: False

    # Test check_in
    print(hotel.check_in('single', 2, 'guest 1'))  # Output: False
    print(hotel.check_in('single', 1, 'guest 1'))  # Output: True
    print(hotel.booked_rooms)  # Output: {'single': {}}

    # Test check_out
    hotel.check_out('single', 2)
    print(hotel.available_rooms)  # Output: {'single': 7, 'double': 3}
    hotel.check_out('triple', 2)
    print(hotel.available_rooms)  # Output: {'single': 7, 'double': 3, 'triple': 2}

    # Test get_available_rooms
    print(hotel.get_available_rooms('single'))  # Output: 7
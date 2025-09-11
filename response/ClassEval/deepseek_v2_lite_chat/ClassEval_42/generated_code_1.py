from collections import defaultdict

class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = defaultdict(int)
        self.booked_rooms = defaultdict(dict)

    def book_room(self, room_type, room_number, name):
        if room_type not in self.available_rooms:
            return False

        if room_number > self.available_rooms[room_type]:
            return room_number - self.available_rooms[room_type]

        self.available_rooms[room_type] -= room_number
        self.booked_rooms[room_type][name] = room_number
        return True

    def check_in(self, room_type, room_number, name):
        if room_type not in self.booked_rooms or room_number not in self.booked_rooms[room_type]:
            return False

        self.available_rooms[room_type] += self.booked_rooms[room_type][room_number]
        del self.booked_rooms[room_type][room_number]
        return True

    def check_out(self, room_type, room_number):
        if room_type not in self.available_rooms or room_number not in self.booked_rooms[room_type]:
            return False

        self.available_rooms[room_type] -= 1
        if room_number not in self.booked_rooms[room_type]:
            self.booked_rooms[room_type][room_number] = 0
        self.booked_rooms[room_type][room_number] -= 1
        return True

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]

# Test cases
if __name__ == "__main__":
    hotel = Hotel('peace hotel', {'single': 5, 'double': 3})
    print(hotel.name)  # Output: peace hotel
    print(hotel.available_rooms)  # Output: defaultdict(<class 'int'>, {'single': 5, 'double': 3})
    print(hotel.booked_rooms)  # Output: defaultdict(<class 'dict'>, {'single': {}, 'double': {}})

    # Test book_room
    print(hotel.book_room('single', 1, 'guest 1'))  # Output: Success!
    print(hotel.book_room('single', 5, 'guest 1'))  # Output: 4
    print(hotel.book_room('single', 4, 'guest 1'))  # Output: Success!
    print(hotel.book_room('single', 1, 'guest 1'))  # Output: False
    print(hotel.book_room('triple', 1, 'guest 1'))  # Output: False

    # Test check_in
    print(hotel.check_in('single', 2, 'guest 1'))  # Output: False

    # Test check_out
    print(hotel.check_out('single', 2))  # Output: True
    print(hotel.available_rooms)  # Output: defaultdict(<class 'int'>, {'single': 7, 'double': 3})
    print(hotel.check_out('triple', 2))  # Output: True
    print(hotel.available_rooms)  # Output: defaultdict(<class 'int'>, {'single': 7, 'double': 3, 'triple': 2})

    # Test get_available_rooms
    print(hotel.get_available_rooms('single'))  # Output: 5
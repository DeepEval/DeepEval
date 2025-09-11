class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def book_room(self, room_type, room_number, name):
        if room_number <= self.available_rooms[room_type]:
            self.available_rooms[room_type] -= room_number
            if room_type not in self.booked_rooms:
                self.booked_rooms[room_type] = {}
            self.booked_rooms[room_type][name] = room_number
            return "Success!"
        else:
            return False

    def check_in(self, room_type, room_number, name):
        if room_type not in self.booked_rooms:
            return False
        if room_number > self.booked_rooms[room_type][name]:
            return False
        self.booked_rooms[room_type][name] -= room_number
        if self.booked_rooms[room_type][name] == 0:
            del self.booked_rooms[room_type][name]
        return True

    def check_out(self, room_type, room_number):
        if room_type not in self.booked_rooms:
            return False
        if room_number > self.booked_rooms[room_type]:
            return False
        self.available_rooms[room_type] += room_number
        self.booked_rooms[room_type] -= room_number
        return True

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]

if __name__ == "__main__":
    hotel = Hotel("Peace Hotel", {"single": 5, "double": 3})
    print(hotel.book_room("single", 1, "Guest 1"))
    print(hotel.book_room("single", 5, "Guest 1"))
    print(hotel.book_room("single", 4, "Guest 1"))
    print(hotel.book_room("single", 1, "Guest 1"))
    print(hotel.book_room("triple", 1, "Guest 1"))

    print(hotel.check_in("single", 2, "Guest 1"))
    print(hotel.check_in("single", 1, "Guest 1"))

    print(hotel.check_out("single", 2))
    print(hotel.check_out("triple", 2))

    print(hotel.get_available_rooms("single"))
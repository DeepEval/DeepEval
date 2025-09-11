import datetime
import numpy as np

class MovieBookingSystem:
    def __init__(self):
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        self.movies.append({
            "name": name,
            "price": price,
            "start_time": datetime.datetime.strptime(start_time, "%H:%M"),
            "end_time": datetime.datetime.strptime(end_time, "%H:%M"),
            "seats": np.zeros((n, n))
        })

    def book_ticket(self, name, seats_to_book):
        movie = None
        for m in self.movies:
            if m["name"] == name:
                movie = m
                break
        if not movie:
            return "Movie not found."

        # Check if the seats are available
        for row, col in seats_to_book:
            if movie["seats"][row][col] == 1:
                return "Booking failed."

        # Book the seats
        for row, col in seats_to_book:
            movie["seats"][row][col] = 1

        return "Booking success."

    def available_movies(self, start_time, end_time):
        return [
            m["name"] for m in self.movies
            if start_time < m["end_time"] and end_time > m["start_time"]
        ]

if __name__ == "__main__":
    system = MovieBookingSystem()
    system.add_movie("Batman", 49.9, "17:05", "19:25", 3)
    print(system.book_ticket("Batman", [(0, 0), (0, 1)]))
    start = datetime.datetime.strptime("12:00", "%H:%M")
    end = datetime.datetime.strptime("22:00", "%H:%M")
    print(system.available_movies(start, end))
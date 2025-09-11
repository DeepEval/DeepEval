from datetime import datetime
import numpy as np

class MovieBookingSystem:
    """
    This is a class for a movie booking system, which allows adding movies, booking tickets, 
    and checking the available movies within a given time range.
    """

    def __init__(self):
        """
        Initialize movies containing the information about movies.
        """
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        """
        Add a new movie into self.movies.
        :param name: str, movie name
        :param price: float, price for one ticket
        :param start_time: str, start time in HH:MM format
        :param end_time: str, end time in HH:MM format
        :param n: int, the size of seats (n*n)
        """
        start_dt = datetime.strptime(start_time, '%H:%M')
        end_dt = datetime.strptime(end_time, '%H:%M')
        seats = np.zeros((n, n))
        movie = {
            'name': name,
            'price': price,
            'start_time': start_dt,
            'end_time': end_dt,
            'seats': seats
        }
        self.movies.append(movie)

    def book_ticket(self, name, seats_to_book):
        """
        Book tickets for a movie. Change the seats value in self.movies if booking is successful.
        :param name: str, movie name
        :param seats_to_book: list of tuples, representing seats to book [(row1, col1), (row2, col2), ...]
        :return: str, booking status message.
        """
        for movie in self.movies:
            if movie['name'].lower() == name.lower():
                seats = movie['seats']
                for seat in seats_to_book:
                    if seats[seat] == 1:
                        return "Booking failed."
                for seat in seats_to_book:
                    seats[seat] = 1
                return "Booking success."
        return "Movie not found."

    def available_movies(self, start_time, end_time):
        """
        Get a list of available movies within the specified time range.
        :param start_time: str, start time in HH:MM format
        :param end_time: str, end time in HH:MM format
        :return: list of str, names of available movies
        """
        start_dt = datetime.strptime(start_time, '%H:%M')
        end_dt = datetime.strptime(end_time, '%H:%M')
        available_movies = []
        for movie in self.movies:
            if movie['start_time'] >= start_dt and movie['end_time'] <= end_dt:
                available_movies.append(movie['name'])
        return available_movies

# Test cases for each method
if __name__ == "__main__":
    system = MovieBookingSystem()

    # Test case for add_movie
    system.add_movie('Batman', 49.9, '17:05', '19:25', 3)
    print(system.movies)

    # Test case for book_ticket
    print(system.book_ticket('Batman', [(0, 0), (0, 1)]))  # Expected: 'Booking success.'
    print(system.book_ticket('Batman', [(0, 0)]))          # Expected: 'Booking failed.'
    print(system.book_ticket('batman', [(0, 0)]))          # Expected: 'Movie not found.'

    # Test case for available_movies
    print(system.available_movies('12:00', '22:00'))       # Expected: ['Batman']
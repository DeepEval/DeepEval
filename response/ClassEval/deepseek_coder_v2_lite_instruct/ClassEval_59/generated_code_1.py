from datetime import datetime
import numpy as np

class MovieBookingSystem:
    """
    This is a class as a movie booking system, which allows to add movies, book tickets and check the available movies within a given time range.
    """

    def __init__(self):
        """
        Initialize movies contains the information about movies
        """
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        """
        Add a new movie into self.movies
        :param name: str, movie name
        :param price: float, price for one ticket
        :param start_time: str
        :param end_time: str
        :param n: int, the size of seats(n*n)
        """
        # Convert start_time and end_time to datetime objects
        start_time = datetime.strptime(start_time, "%H:%M")
        end_time = datetime.strptime(end_time, "%H:%M")
        # Create a 2D array of seats initialized to 0
        seats = np.zeros((n, n))
        # Add the movie to the list of movies
        self.movies.append({
            'name': name,
            'price': price,
            'start_time': start_time,
            'end_time': end_time,
            'seats': seats
        })

    def book_ticket(self, name, seats_to_book):
        """
        Book tickets for a movie. Change the seats value in self.movies if book successfully.
        :param name: str, movie name
        :param seats_to_book: list of tuples, representing seats to book [(row1, col1), (row2, col2), ...]
        :return: str, booking status message. "Movie not found." for no such movie.
                "Booking success." for successfully booking, or "Booking failed." otherwise
        """
        # Find the movie by name
        movie = next((movie for movie in self.movies if movie['name'].lower() == name.lower()), None)
        if movie is None:
            return "Movie not found."
        
        # Check if seats are available
        seats = movie['seats']
        for row, col in seats_to_book:
            if row < 0 or row >= seats.shape[0] or col < 0 or col >= seats.shape[1] or seats[row, col] == 1:
                return "Booking failed."
            seats[row, col] = 1
        
        return "Booking success."

    def available_movies(self, start_time, end_time):
        """
        Get a list of available movies within the specified time range
        :param start_time: str, start time in HH:MM format
        :param end_time: str, end time in HH:MM format
        :return: list of str, names of available movies
        """
        # Convert start_time and end_time to datetime objects
        start_time = datetime.strptime(start_time, "%H:%M")
        end_time = datetime.strptime(end_time, "%H:%M")
        # Get the list of available movies
        available_movies = [movie['name'] for movie in self.movies if start_time <= movie['start_time'] < end_time and end_time <= movie['end_time']]
        return available_movies

# Test cases
if __name__ == "__main__":
    system = MovieBookingSystem()
    # Add a movie
    system.add_movie('Batman', 49.9, '17:05', '19:25', 3)
    # Book a ticket
    print(system.book_ticket('Batman', [(0, 0), (0, 1)]))  # Booking success.
    print(system.book_ticket('Batman', [(0, 0)]))         # Booking failed.
    print(system.book_ticket('batman', [(0, 0)]))         # Movie not found.
    # Available movies
    print(system.available_movies('12:00', '22:00'))     # ['Batman']
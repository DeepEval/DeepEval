from datetime import datetime
import numpy as np

class MovieBookingSystem:
    """
    A class for a movie booking system.
    """

    def __init__(self):
        """
        Initialize movies contains the information about movies
        """
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        """
        Add a new movie into self.movies
        """
        self.movies.append({
            'name': name,
            'price': price,
            'start_time': datetime.strptime(start_time, '%H:%M'),
            'end_time': datetime.strptime(end_time, '%H:%M'),
            'seats': np.zeros((n, n), dtype=bool)  # Initialize all seats to False (not booked)
        })

    def book_ticket(self, name, seats_to_book):
        """
        Book tickets for a movie. Change the seats value in self.movies if book successfully.
        """
        for movie in self.movies:
            if movie['name'] == name:
                for row, col in seats_to_book:
                    if 0 <= row < movie['seats'].shape[0] and 0 <= col < movie['seats'].shape[1]:
                        if not movie['seats'][row, col]:
                            movie['seats'][row, col] = True  # Seat is booked
                            return "Booking success."
                        else:
                            return "Booking failed."
                return "Movie not found."

    def available_movies(self, start_time, end_time):
        """
        Get a list of available movies within the specified time range
        """
        available_movies = []
        for movie in self.movies:
            if movie['start_time'] >= datetime.strptime(start_time, '%H:%M') and movie['end_time'] <= datetime.strptime(end_time, '%H:%M'):
                if np.sum(movie['seats']) == 0:  # All seats are available
                    available_movies.append(movie['name'])
        return available_movies

# Example usage:
if __name__ == "__main__":
    system = MovieBookingSystem()
    system.add_movie('Batman', 49.9, '17:05', '19:25', 3)
    print(system.book_ticket('Batman', [(0, 0), (0, 1)]))  # Should print 'Booking success.'
    print(system.book_ticket('Batman', [(0, 0)]))  # Should print 'Booking failed.'
    print(system.book_ticket('Batman', [(0, 0), (1, 1)]))  # Should print 'Movie not found.'
    print(system.available_movies('12:00', '22:00'))  # Should print ['Batman']
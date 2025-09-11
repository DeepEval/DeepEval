from datetime import datetime
import numpy as np

class MovieBookingSystem:
    """
    This is a class as movie booking system, which allows to add movies, book tickets 
    and check the available movies within a given time range. 
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
        start_time_dt = datetime.strptime(start_time, '%H:%M')
        end_time_dt = datetime.strptime(end_time, '%H:%M')
        seats = np.zeros((n, n))
        movie = {
            'name': name,
            'price': price,
            'start_time': start_time_dt,
            'end_time': end_time_dt,
            'seats': seats
        }
        self.movies.append(movie)

    def book_ticket(self, name, seats_to_book):
        """
        Book tickets for a movie. Change the seats value in self.movies if booked successfully.
        :param name: str, movie name
        :param seats_to_book: list of tuples, representing seats to book [(row1, col1), (row2, col2), ...]
        :return: str, booking status message
        """
        for movie in self.movies:
            if movie['name'].lower() == name.lower():
                for seat in seats_to_book:
                    row, col = seat
                    if movie['seats'][row][col] == 0:  # Seat is available
                        movie['seats'][row][col] = 1  # Mark seat as booked
                    else:
                        return "Booking failed."
                return "Booking success."
        return "Movie not found."

    def available_movies(self, start_time, end_time):
        """
        Get a list of available movies within the specified time range
        :param start_time: str, start time in HH:MM format
        :param end_time: str, end time in HH:MM format
        :return: list of str, names of available movies
        """
        start_time_dt = datetime.strptime(start_time, '%H:%M')
        end_time_dt = datetime.strptime(end_time, '%H:%M')
        available = []
        for movie in self.movies:
            if (movie['start_time'] >= start_time_dt and movie['start_time'] < end_time_dt) or \
               (movie['end_time'] > start_time_dt and movie['end_time'] <= end_time_dt):
                available.append(movie['name'])
        return available

# Testing the MovieBookingSystem class
if __name__ == "__main__":
    system = MovieBookingSystem()
    
    # Test case for adding a movie
    system.add_movie('Batman', 49.9, '17:05', '19:25', 3)
    print(system.movies)  # Should print the movie details
    
    # Test case for booking tickets
    output = system.book_ticket('Batman', [(0, 0), (0, 1)])
    print(output)  # Should print 'Booking success.'
    
    output = system.book_ticket('Batman', [(0, 0)])
    print(output)  # Should print 'Booking failed.'
    
    output = system.book_ticket('batman', [(0, 0)])
    print(output)  # Should print 'Movie not found.'
    
    # Test case for available movies
    available = system.available_movies('12:00', '22:00')
    print(available)  # Should print ['Batman']
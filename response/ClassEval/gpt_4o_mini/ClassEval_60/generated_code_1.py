import sqlite3

class MovieTicketDB:
    """
    This is a class for movie database operations, which allows for inserting movie information, searching for movie information by name, and deleting movie information by name.
    """

    def __init__(self, db_name):
        """
        Initializes the MovieTicketDB object with the specified database name.
        :param db_name: str, the name of the SQLite database.
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """
        Creates a "tickets" table in the database if it does not exist already.
        :return: None
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_name TEXT NOT NULL,
                theater_name TEXT NOT NULL,
                seat_number TEXT NOT NULL,
                customer_name TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        """
        Inserts a new ticket into the "tickets" table.
        :param movie_name: str, the name of the movie.
        :param theater_name: str, the name of the theater.
        :param seat_number: str, the seat number.
        :param customer_name: str, the name of the customer.
        :return: None
        """
        self.cursor.execute('''
            INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
            VALUES (?, ?, ?, ?)
        ''', (movie_name, theater_name, seat_number, customer_name))
        self.connection.commit()

    def search_tickets_by_customer(self, customer_name):
        """
        Searches for tickets in the "tickets" table by customer name.
        :param customer_name: str, the name of the customer to search for.
        :return: list of tuples, the rows from the "tickets" table that match the search criteria.
        """
        self.cursor.execute('''
            SELECT * FROM tickets WHERE customer_name = ?
        ''', (customer_name,))
        return self.cursor.fetchall()

    def delete_ticket(self, ticket_id):
        """
        Deletes a ticket from the "tickets" table by ticket ID.
        :param ticket_id: int, the ID of the ticket to delete.
        :return: None
        """
        self.cursor.execute('''
            DELETE FROM tickets WHERE id = ?
        ''', (ticket_id,))
        self.connection.commit()

    def close(self):
        """Closes the database connection."""
        self.connection.close()

# Test cases
if __name__ == "__main__":
    # Create a database instance
    ticket_db = MovieTicketDB("ticket_database.db")
    
    # Test create_table
    ticket_db.create_table()
    
    # Test insert_ticket
    ticket_db.insert_ticket("Movie A", "Theater 1", "A1", "John Doe")
    ticket_db.insert_ticket("Movie B", "Theater 2", "B2", "Jane Smith")
    
    # Test search_tickets_by_customer
    result = ticket_db.search_tickets_by_customer("John Doe")
    print(f"Search result for John Doe: {result}")  # Should return 1 record for John Doe
    
    # Test delete_ticket
    ticket_db.delete_ticket(result[0][0])  # Deletes the first ticket (ID of John Doe's ticket)
    
    # Verify deletion
    result_after_delete = ticket_db.search_tickets_by_customer("John Doe")
    print(f"Search result for John Doe after deletion: {result_after_delete}")  # Should return an empty list
    
    # Close the database connection
    ticket_db.close()
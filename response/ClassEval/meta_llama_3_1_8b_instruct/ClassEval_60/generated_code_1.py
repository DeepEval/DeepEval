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
        Fields include ID of type int, movie name of type str, theater name of type str, seat number of type str, and customer name of type str
        :return: None
        """
        query = """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY,
                movie_name TEXT NOT NULL,
                theater_name TEXT NOT NULL,
                seat_number TEXT NOT NULL,
                customer_name TEXT NOT NULL
            )
        """
        self.cursor.execute(query)
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
        query = """
            INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
            VALUES (?,?,?,?)
        """
        self.cursor.execute(query, (movie_name, theater_name, seat_number, customer_name))
        self.connection.commit()


    def search_tickets_by_customer(self, customer_name):
        """
        Searches for tickets in the "tickets" table by customer name.
        :param customer_name: str, the name of the customer to search for.
        :return: list of tuples, the rows from the "tickets" table that match the search criteria.
        """
        query = """
            SELECT * FROM tickets
            WHERE customer_name =?
        """
        self.cursor.execute(query, (customer_name,))
        return self.cursor.fetchall()


    def delete_ticket(self, ticket_id):
        """
        Deletes a ticket from the "tickets" table by ticket ID.
        :param ticket_id: int, the ID of the ticket to delete.
        :return: None
        """
        query = """
            DELETE FROM tickets
            WHERE id =?
        """
        self.cursor.execute(query, (ticket_id,))
        self.connection.commit()


if __name__ == "__main__":
    ticket_db = MovieTicketDB("ticket_database.db")
    # Create table
    ticket_db.create_table()

    # Insert ticket
    ticket_db.insert_ticket("Movie A", "Theater 1", "A1", "John Doe")

    # Search tickets by customer
    result = ticket_db.search_tickets_by_customer("John Doe")
    print("Search result:", result)

    # Delete ticket
    ticket_db.delete_ticket(1)

    # Search tickets by customer (after deletion)
    result = ticket_db.search_tickets_by_customer("John Doe")
    print("Search result after deletion:", result)
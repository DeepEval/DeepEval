import sqlite3

class MovieTicketDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS tickets (ID INTEGER PRIMARY KEY, movie_name TEXT, theater_name TEXT, seat_number TEXT, customer_name TEXT)")
        self.connection.commit()

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        self.cursor.execute("INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name) VALUES (?, ?, ?, ?)", (movie_name, theater_name, seat_number, customer_name))
        self.connection.commit()

    def search_tickets_by_customer(self, customer_name):
        self.cursor.execute("SELECT * FROM tickets WHERE customer_name = ?", (customer_name,))
        result = self.cursor.fetchall()
        return result

    def delete_ticket(self, ticket_id):
        self.cursor.execute("DELETE FROM tickets WHERE ID = ?", (ticket_id,))
        self.connection.commit()

if __name__ == "__main__":
    ticket_db = MovieTicketDB("ticket_database.db")
    ticket_db.create_table()
    ticket_db.insert_ticket("Movie A", "Theater 1", "A1", "John Doe")
    result = ticket_db.search_tickets_by_customer("John Doe")
    print(result)
    ticket_db.delete_ticket(1)
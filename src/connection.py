import sqlite3

class Database:
    def __init__(self, path: str):
        self.conn = sqlite3.connect('db.sqlite')
        self.cur = conn.cursor()

    def create_table_if_not_exist(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                password TEXT
            )
        """)
        self.conn.commit()

    def query(self, query: str, val: tuple, get_output = False):
        self.cur.execute(query, val)
        self.conn.commit()
        if get_output:
            return self.cur.fetchall()
        return None

    def close():
        self.conn.close()

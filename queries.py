import sqlite3
from sqlite3 import Error
import pickle
import pandas as pd

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

if __name__ == '__main__':
    
    database = "db.sqlite3"
    con = sqlite3.connect(database)
    cursor = con.cursor()

    # Get all Existing Tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())

    # Get all Existing Tables from bluedivis project
    cursor.execute("SELECT * FROM bluedivis_dweet")
    names = list(map(lambda x: x[0], cursor.description))
    print(names)
    print(cursor.fetchall())

    cursor.execute("DROP TABLE IF EXISTS bluedivis_mlresult")
    table = """ CREATE TABLE bluedivis_mlresult (
                id INT,
                TICKER CHAR(255),
                DIVSPAID INT,
                DIVYIELD REAL,
                CLUSTER INT
            ); """
    cursor.execute(table)

    # Put ML Result into sqlite DB
    df_MLResult = pickle.load(open("df_all.pkl", "rb"))
    df_MLResult["TICKER"] = df_MLResult["TICKER"].str.split("-", n = 1, expand = True)[0]
    df_MLResult.reset_index(inplace=True)
    df_MLResult.rename(columns={"index" : "id"}, inplace=True)
    df_MLResult[["id", "TICKER", "divs_paid", "div_yield_mean", "cluster"]].to_sql("bluedivis_mlresult", con, if_exists="replace", index=False)

    
    cursor.execute("SELECT * FROM bluedivis_mlresult WHERE TICKER = 'ZION'")
    print(cursor.fetchall())

    cursor.execute("SELECT * FROM bluedivis_mlresult")
    names = list(map(lambda x: x[0], cursor.description))
    print(names)
    print(cursor.fetchall())


    # sql = ''' INSERT INTO MLRESULTS(TICKER,Score)
    #           VALUES(?,?) '''
    # task_1 = ("Test", 1)
    # task_2 = ("Test1", 0)
    # task_3 = ("Test2", 0)

    # cursor.execute(sql, task_1)
    # cursor.execute(sql, task_2)
    # cursor.execute(sql, task_3)





     
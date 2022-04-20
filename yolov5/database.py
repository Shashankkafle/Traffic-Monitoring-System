import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="12345",
  database="stats"
)

myCursor = mydb.cursor()

myCursor.execute("select * from test")

for i in myCursor:
  print (i)
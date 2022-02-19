import mysql.connector
from PIL import Image

image = Image.open('C:/Users/Acer/Desktop/Screenshot 2021-11-08 120800.jpg')
speed = 70

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

photo = convertToBinaryData("C:/Users/Acer/Desktop/ACE074BCT070.jpg")

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="12345",
  database="stats"
)

mycursor = mydb.cursor()

def save_data(image,speed):
    photo = image.tobytes()
    sql = "INSERT INTO test (image, speed) VALUES (%s, %s)"
    val = (photo, speed)
    mycursor.execute(sql, val)
    mydb.commit()
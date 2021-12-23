from PIL import Image
def get_speed(vehicle):
     print("speed in update2",vehicle.speed ,"of ",vehicle.id)
     file = open("Speed_Data.txt", "a")
     file.write("id:"+str(vehicle.id)+"speed"+str(vehicle.speed)+"\n")

    
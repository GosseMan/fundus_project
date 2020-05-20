import os
from glob import glob
from PIL import Image
def resize_dir():
    path = '/home/psj/Desktop/FUNDUS_DATA/fundus_data_binary'
    dir_list = os.listdir(path)

    newpath = '/home/psj/Desktop/FUNDUS_DATA_480'

    for dir in dir_list:
        dir_path = path+'/'+dir
        file_list = os.listdir(dir_path)
        newpath_dir = newpath +'/fundus_data_binary/'+dir
        try:
            if not os.path.exists(newpath_dir):
                os.makedirs(newpath_dir)
        except OSError:
            print('Error: creating '+ newpath_dir)
        for file in file_list:
            image = Image.open(dir_path+'/'+file)
            resize_image = image.resize((480,480))
            resize_image.save(newpath_dir+'/'+file,"JPEG",quality=100)

def resize_file():
    path = '/home/psj/Desktop/FUNDUS_DATA/all_files'
    file_list = os.listdir(path)
    newpath = '/home/psj/Desktop/FUNDUS_DATA_480/all_files'
    try:
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    except OSError:
        print('Error: creating '+ newpath)
    for file in file_list:
        image = Image.open(path+'/'+file)
        resize_image = image.resize((480,480))
        resize_image.save(newpath+'/'+file,"JPEG",quality=100)

if __name__ == "__main__":
    resize_file()

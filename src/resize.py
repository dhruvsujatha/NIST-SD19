from PIL import Image as im
import numpy as np
folder_path = "C:/NN/Combined/EnglishFnt/English/Dataset 4/Img/"
saving_folder_path = "C:/NN/Dataset 4/"
main_iteration = 0
pixels = 50

for i in range(62):
    p = np.round((i / 61) * 100, 2)
    print('\r''Data loading:', p, '%', end='')
    folder_name = folder_path  #  + "Sample" + str(i + 1).zfill(3) + "/"
    for j in range(55):
        filename = folder_name + "img" + str(i + 1).zfill(3) + "-" + str(j + 1).zfill(3) + ".png"
        image = im.open(filename)
        image = image.resize((pixels, pixels))
        saving_filename = saving_folder_path + "img" + str(i + 1).zfill(2) + "-" + str(j + 1017).zfill(4) + ".png"
        image.save(saving_filename)
        image.close()



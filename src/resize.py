from PIL import Image as im

folder_path = "C:/NN/NIST SD 19/by_class/"
saving_folder_path = "C:/NN/NIST SD 19/DataSet/"
main_iteration = 0
pixels = 50

for i in range(47, 122):
    print(main_iteration)
    if 47 < i < 58 or 64 < i < 91 or 96 < i < 123:
        to_get_folder_path = folder_path + str(i) + "/train_" + str(hex(i))[2:4]
        for j in range(400):
            filename = to_get_folder_path + "/train_" + str(hex(i))[2:4] + "_" + format(j, '0' + str(5)) + ".png"
            image = im.open(filename)
            image = image.resize((pixels, pixels))
            saving_filename = saving_folder_path + str(i) + "-" + str(j) + ".png"
            image.save(saving_filename)
    main_iteration = main_iteration + 1

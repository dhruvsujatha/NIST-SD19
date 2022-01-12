import codecs
import os
import shutil

path = "/Users/dsujatha/NIST/by_class/"
files = os.listdir(path)
print(files)
for f in files:
    str_bin = codecs.decode(f, 'hex')
    shutil.move(path + f ,path + str(str_bin, 'utf-8'))

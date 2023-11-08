import os
import random
import io
import csv 

# Программа конвертации данных из txt файла csv - таблицу
def convert_to_fbx(input, output): # "data/OxfordPets/test.txt", 'data/petfaces/val.csv'
    input_f = open(input, mode="r", encoding="utf-8")
    input_list = input_f.readlines()
    header = ['path', 'multiclass', 'class_bin','class_inner']
    
    with open(output, 'w', newline='') as output_f:
            writer = csv.writer(output_f)
            
            # заголовки
            writer.writerow(header)

            for i in input_list:
                string = i.split(" ")
                x = int(string[3])-1 if string[2] == "1" else 11 + int(string[3])
                data = ["images/"+string[0]+".jpg", x, abs(int(string[2])-2), int(string[3])-1]
                # данные                
                writer.writerow(data)

def train_val_split(path, file, split = 0.8, t = "train.csv", v = "val.csv"):
    f = open(os.path.join(path, file))
    train = open(os.path.join(path, t), "w")
    val = open(os.path.join(path, v), "w")
    h = f.readline()
    train.write(h)
    val.write(h)
    for t in f.readlines():
        if random.random() < split:
            train.write(t)
        else:
            val.write(t)
    f.close()
    train.close()
    val.close()

def train_test_split_for_direct_ty_file(path, split=0.8):
    train = open(os.path.join(path, "train.csv"), "w")
    val = open(os.path.join(path, "val.csv"), "w")
    train.write("path,multiclass,binaryclass,multiclassName,binaryclassName\n")
    val.write("path,multiclass,binaryclass,multiclassName,binaryclassName\n")
    classes = os.listdir(path)
    for clas in classes:
        if len(clas.split(".")) == 1:
            image_paths = os.listdir(os.path.join(path, clas))
            for image_path in image_paths:
                if random.random() < split:
                    train.write(
                        f"{os.path.join(clas, image_path)},{ classes.index(clas)},{int(clas.split('_')[0] == 'cat')},{clas},{clas.split('_')[0]}\n"
                    )
                else:
                    val.write(
                        f"{os.path.join(clas, image_path)},{ classes.index(clas)},{int(clas.split('_')[0] == 'cat')},{clas},{clas.split('_')[0]}\n"
                    )
    train.close()
    val.close()

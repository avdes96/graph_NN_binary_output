# Taken from https://github.com/qiankunmu/HDHGN

import ast
import javalang
import javalang.parse
import os
from sklearn.model_selection import train_test_split
import random

random.seed(42)

def get_valid_files(base_folder, sub_folder_string, outcome, files_paths, labels):
    folder = os.path.join(base_folder, sub_folder_string)
    files = os.listdir(folder)
    for file_name in files:
        if file_name.endswith('.py'):
            file_path = os.path.join(folder, file_name)
            code = open(file_path, encoding='utf-8').read()
            try:
                ast.parse(code)
                files_paths.append(file_path)
                labels.append(outcome)
            except SyntaxError:
                pass

def shuffle_text_file(filename):
    lines = open(filename).readlines()
    random.shuffle(lines)
    open(filename, 'w').writelines(lines)


def split_data():
    questions = ["question_1", "question_2", "question_3", "question_4", "question_5"]
    files_paths = []
    labels = []
    #cwd = os.getcwd()
    data_folder = "data"
    for question in questions:
        code_folder = os.path.join(".", data_folder, question, "code")
        get_valid_files(code_folder, "correct", 1, files_paths, labels)
        get_valid_files(code_folder, "fail", 0, files_paths, labels)
        get_valid_files(code_folder, "wrong", 0, files_paths, labels)

    train_files_paths, vt_files_paths, train_labels, vt_labels = train_test_split(files_paths, labels, test_size=0.4,
                                                                                  stratify=labels)
    valid_files_paths, test_files_paths, valid_labels, test_labels = train_test_split(vt_files_paths, vt_labels,
                                                                                      test_size=0.5, stratify=vt_labels)
    train_file = open("data/train_files_paths.txt", "w+")
    valid_file = open("data/valid_files_paths.txt", "w+")
    test_file = open("data/test_files_paths.txt", "w+")
    for train_file_path in train_files_paths:
        train_file.write(train_file_path)
        train_file.write("\n")
    train_file.close()
    shuffle_text_file("data/train_files_paths.txt")


    for valid_file_path in valid_files_paths:
        valid_file.write(valid_file_path)
        valid_file.write("\n")
    valid_file.close()
    shuffle_text_file("data/valid_files_paths.txt")

    for test_file_path in test_files_paths:
        test_file.write(test_file_path)
        test_file.write("\n")    
    test_file.close()
    shuffle_text_file("data/test_files_paths.txt")
    
    print("finish")

def splitdata_java():
    dir_path = "data/Project_CodeNet_Java250"

    files_paths = []
    labels = []
    for root, dir, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith('.java'):
                file_path = os.path.join(root, file_name)
                code = open(file_path, encoding='utf-8').read()
                try:
                    javalang.parse.parse(code)
                    files_paths.append(file_path)
                    labels.append(root)
                except javalang.tokenizer.LexerError:
                    pass

    train_files_paths, vt_files_paths, train_labels, vt_labels = train_test_split(files_paths, labels, test_size=0.4,
                                                                                  stratify=labels)
    valid_files_paths, test_files_paths, valid_labels, test_labels = train_test_split(vt_files_paths, vt_labels,
                                                                                      test_size=0.5, stratify=vt_labels)
    train_file = open("data/train_files_paths_java.txt", "w+")
    valid_file = open("data/valid_files_paths_java.txt", "w+")
    test_file = open("data/test_files_paths_java.txt", "w+")
    for train_file_path in train_files_paths:
        train_file.write(train_file_path)
        train_file.write("\n")
    train_file.close()

    for valid_file_path in valid_files_paths:
        valid_file.write(valid_file_path)
        valid_file.write("\n")
    valid_file.close()

    for test_file_path in test_files_paths:
        test_file.write(test_file_path)
        test_file.write("\n")
    test_file.close()

    print("finish")

if __name__ == '__main__':
    for filename in ["train_files_paths_java", "valid_files_paths_java", "test_files_paths_java"]:
        filename = "data/" + filename + ".txt"
        try:
            os.remove(filename)
        except OSError:
            pass
    print("start spliting python data")
    split_data()
    # print("start spliting java data")
    # splitdata_java()

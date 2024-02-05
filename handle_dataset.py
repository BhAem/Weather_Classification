import os
import shutil
import random

# # 设置源文件夹和目标文件夹
# source_folder = "./MWD/test_old"
# destination_folder = "./MWD/test"
#
# # 确保目标文件夹存在
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
#
# # 获取源文件夹中的所有文件
# file_list = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
#
# # 如果文件数量少于1568，就全部复制
# if len(file_list) < 1568:
#     files_to_copy = file_list
# else:
#     # 随机选择1568个文件
#     files_to_copy = random.sample(file_list, 1568)
#
# # 复制文件到目标文件夹
# for file_name in files_to_copy:
#     # 构建完整的文件路径
#     src_path = os.path.join(source_folder, file_name)
#     dst_path = os.path.join(destination_folder, file_name)
#
#     # 复制文件
#     shutil.copy(src_path, dst_path)
#     print(f"Copied {file_name} to {destination_folder}")
#
# print("All files have been copied.")



# test_path = "/home/Huangwb/Wuqf/classify/MWD/test3"
# imgs = os.listdir(test_path)
# a,b,c,d,e=0,0,0,0,0
# for idx, img in enumerate(imgs):
#     # path = os.path.join(test_path, img)
#     kind = img.split(".")[0].split("_")[-1]
#     if kind == "day":
#         a += 1
#     if kind == "rain":
#         b += 1
#     if kind == "snow":
#         c += 1
#     if kind == "fog":
#         d += 1
#     if kind == "night":
#         e += 1
# print(a,b,c,d,e)





ori_path = "/home/Huangwb/Wuqf/classify/MWD/test_old"
test_path = "/home/Huangwb/Wuqf/classify/MWD/test"
test2_path = "/home/Huangwb/Wuqf/classify/MWD/test3"
imgs = os.listdir(test_path)
a,b,c,d,e=0,0,0,0,0
for idx, img in enumerate(imgs):
    # path = os.path.join(test_path, img)
    kind = img.split(".")[0].split("_")[-1]
    if kind == "snow" and c < 270:
        c += 1
        arr = img.split(".")[0].split("_")[:-1]
        dayname,rainname,fogname,nightname="","","",""
        newname = ""
        for j, ss in enumerate(arr):
            newname += ss
            newname += "_"
        dayname = newname + "day.jpg"
        rainname = newname + "rain.jpg"
        fogname = newname + "fog.jpg"
        nightname = newname + "night.jpg"
        if not os.path.exists(os.path.join(test_path, dayname)):
            shutil.copy(os.path.join(ori_path, dayname), os.path.join(test2_path, dayname))
        elif not os.path.exists(os.path.join(test_path, fogname)):
            shutil.copy(os.path.join(ori_path, fogname), os.path.join(test2_path, fogname))
        elif not os.path.exists(os.path.join(test_path, nightname)):
            shutil.copy(os.path.join(ori_path, nightname), os.path.join(test2_path, nightname))
        elif not os.path.exists(os.path.join(test_path, rainname)):
            shutil.copy(os.path.join(ori_path, rainname), os.path.join(test2_path, rainname))
    elif kind == "rain" and b < 260:
        b += 1
        arr = img.split(".")[0].split("_")[:-1]
        dayname,rainname,fogname,nightname="","","",""
        newname = ""
        for j, ss in enumerate(arr):
            newname += ss
            newname += "_"
        dayname = newname + "day.jpg"
        rainname = newname + "snow.jpg"
        fogname = newname + "fog.jpg"
        nightname = newname + "night.jpg"
        if not os.path.exists(os.path.join(test_path, dayname)):
            shutil.copy(os.path.join(ori_path, dayname), os.path.join(test2_path, dayname))
        elif not os.path.exists(os.path.join(test_path, fogname)):
            shutil.copy(os.path.join(ori_path, fogname), os.path.join(test2_path, fogname))
        elif not os.path.exists(os.path.join(test_path, nightname)):
            shutil.copy(os.path.join(ori_path, nightname), os.path.join(test2_path, nightname))
        elif not os.path.exists(os.path.join(test_path, rainname)):
            shutil.copy(os.path.join(ori_path, rainname), os.path.join(test2_path, rainname))
    else:
        shutil.copy(os.path.join(test_path, img), os.path.join(test2_path, img))
    print(img)

print(a,b,c,d,e)

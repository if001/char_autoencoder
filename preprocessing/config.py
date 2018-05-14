import os


class Config():
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    last1 = run_dir_path.split("/")[-1]
    last2 = run_dir_path.split("/")[-2]
    up_dir = run_dir_path.replace(last1, "")
    up_dir = up_dir.replace(last2 + "/", "")
    image_dir_path = up_dir + "string2image/image"

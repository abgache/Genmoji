import numpy as np
import os, time
from PIL import Image
from scripts.prompt_enhancer import enhance_prompt
from scripts.time_log import time_log_module as tlm

def format_time(secs):
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    result = ""
    if h > 0:
        result += f"{h}h"
    if m > 0:
        result += f"{m}min"
    if s > 0 or result == "":
        result += f"{s}s"
    return result

class data(): # je voulais faire qlq chose de bien plus complexe mais bref
    def __init__(self):
        self.data_path = "training_data"
        self.len_data = 3770
        self.resolution = (160, 160)
    def load_data(self) -> list:
        self.data = [(np.array(Image.open(f"{self.data_path}\img{i}.png").resize(self.resolution)).astype(np.float16) / 255.0, open(f"{self.data_path}\img{i}.txt", 'r').read()) for i in range(1, self.len_data+1)] # Holy ram usage
        if len(self.data) != self.len_data:
            print(f"{tlm()}-[ALERT] : The number of examples loaded is not equal to the total number of trainning examples!")
    def save_enhanced_prompt(self, overwrite=False):
        if self.data == None:
            raise OSError(f"{tlm()} You have to fist load the data to enhance the prompt, and then load it again.")
        check = 0
        for i, bloc in enumerate(self.data, start=1): # 1940
            if os.path.exists(f"{self.data_path}\\img{i}ai.txt") and not overwrite:
                check += 1
                continue
            tmp = str(bloc[1])
            enhanced = enhance_prompt(tmp, v=False, local=True)
            del tmp
            with open(f"{self.data_path}\\img{i}ai.txt", "x") as file:
                file.write(enhanced)
            check += 1
        if not check == self.len_data:
            print(f"{tlm()}-[ALERT] : The number of enhanced prompt is not equal to the total number of trainning examples!")
            return True
        return True
    def load_enhanced_data(self) -> list:
        if not os.path.exists(f"{self.data_path}\img1ai.txt"):
            raise FileNotFoundError(f"{tlm()} You can't load enhanced prompts if they doesn't exists.")
        self.data = [(np.array(Image.open(f"{self.data_path}\img{i}.png").resize(self.resolution)).astype(np.float16) / 255.0, open(f"{self.data_path}\img{i}ai.txt", 'r').read()) for i in range(1, self.len_data+1)]
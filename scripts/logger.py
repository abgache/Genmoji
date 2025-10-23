from scripts.time_log import time_log_module as tlm, time_log_module_files as tlmf
import time
import requests as r

def webhook_post(content, webhook_url):
    data = {"content": content}
    response = r.post(webhook_url, json=data)
    return response
    
class logger():
    def __init__(self, discord_webhook=""):
        self.start = str(time.time())
        self.actual_log = f"{tlmf()}.log"
        self.discord_webhook = discord_webhook
        try:
            with open(f"logs\{self.actual_log}", "w") as file:
                file.write(f"Start point [{self.start}]\n")
        except FileNotFoundError:
            with open(f"logs\{self.actual_log}", "x") as file:
                file.write(f"Start point [{self.start}]\n")
    def log(self, data, v=True, Wh=True, mention=True):
        if not isinstance(data, str):
            with open(f"logs\{self.actual_log}", "a") as file:
                file.write(f"{tlm()} The input data for the log function is not a string. The data will NOT be logged.\n")
            raise Warning(f"{tlm()} The input data for the log function is not a string. The data will NOT be logged.")
        with open(f"logs\{self.actual_log}", "a") as file:
            file.write(f"{tlm()} {data}\n")
        print(f"{tlm()} {data}")
        if Wh and not self.discord_webhook == "":
            if mention:
                tmp = webhook_post(f"{tlm()} - ||@everyone|| {data}", self.discord_webhook)
            else:
                tmp = webhook_post(f"{tlm()} {data}", self.discord_webhook)

    
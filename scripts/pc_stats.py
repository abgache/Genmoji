import threading
import time
import psutil
import subprocess

def clear_line():
    print("\r", end="")

def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        gpu_usage, vram_used, temp, vram_total = result.stdout.strip().split(", ")
        return int(gpu_usage), int(temp), int(vram_used), int(vram_total)
    except Exception:
        return None, None, None, None

def monitor(stop_event):
    while not stop_event.is_set():
        gpu_usage, temp, vram_used, vram_total = get_gpu_info()
        ram_used = psutil.virtual_memory().used // (1024*1024)
        ram_total = psutil.virtual_memory().total // (1024*1024)

        if gpu_usage is not None:
            info = f"GPU: {gpu_usage}% | Temp: {temp}°C | VRAM: {vram_used}/{vram_total}MB | RAM: {ram_used}/{ram_total}MB"
        else:
            info = f"GPU info not available | RAM: {ram_used}/{ram_total}MB"

        clear_line()
        print(info, end="")
        time.sleep(1)
    clear_line()
    print("Monitoring stopped.                 ")

# Exemple de fonction longue
def long_function():
    for i in range(10):
        time.sleep(2)  # simule du calcul lourd
    print("\nFunction finished!")

if __name__ == "__main__":
    stop_event = threading.Event()
    t = threading.Thread(target=monitor, args=(stop_event,))
    t.start()

    long_function()  # exécution de la fonction

    stop_event.set()  # arrête le monitoring
    t.join()

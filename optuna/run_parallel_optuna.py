import multiprocessing
import subprocess
import os
import time

def run_optuna_script(process_id):
    log_dir = "optuna_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{process_id}.txt")
    with open(log_file, 'w') as f:
        subprocess.run(["diambra", "run", "python3", "optuna_diambra.py", "--study-name", "ppo_study", "--storage", "postgresql://optuna_user:password@localhost/optuna_db", "--process-id", str(process_id)], stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    num_processes = 6  # Number of parallel processes
    processes = []

    for i in range(num_processes):
        p = multiprocessing.Process(target=run_optuna_script, args=(i,))
        p.start()
        processes.append(p)
        time.sleep(5)  # Stagger the start of each process by 5 seconds

    for p in processes:
        p.join()

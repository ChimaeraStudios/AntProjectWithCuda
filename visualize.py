import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.animation import FuncAnimation

def load_files():
    env_files = sorted(glob.glob("cmake-build-debug-visual-studio/environment_*.csv"))
    ants_files = sorted(glob.glob("cmake-build-debug-visual-studio/ants_*.csv"))
    valid_env_files = []
    valid_ants_files = []

    for env_file, ants_file in zip(env_files, ants_files):
        try:
            env_data = np.loadtxt(env_file, delimiter=",")
            ants_data = np.loadtxt(ants_file, delimiter=",", dtype=int)
            valid_env_files.append(env_file)
            valid_ants_files.append(ants_file)
        except Exception as e:
            print(f"Errore nel file {env_file} o {ants_file}: {e}")

    return valid_env_files, valid_ants_files

def visualize_frame(environment_file, ants_file):
    environment = np.loadtxt(environment_file, delimiter=",")
    ants = np.loadtxt(ants_file, delimiter=",", dtype=int)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(environment, cmap="Greys", origin="upper")
    ax.set_title("Ambiente e Formiche")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, environment.shape[1])
    ax.set_ylim(0, environment.shape[0])

    for ant in ants:
        x, y, has_food = ant
        ax.scatter(x, y, c="red" if has_food else "blue", s=5)

    plt.show()

def create_animation(env_files, ants_files):
    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame):
        ax.clear()
        environment = np.loadtxt(env_files[frame], delimiter=",")
        ants = np.loadtxt(ants_files[frame], delimiter=",", dtype=int)

        ax.imshow(environment, cmap="Greys", origin="upper")
        ax.set_title(f"Ambiente e Formiche - Frame {frame + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        for ant in ants:
            x, y, has_food = ant
            ax.scatter(x, y, c="red" if has_food else "blue", s=5)

    ani = FuncAnimation(fig, update, frames=len(env_files), interval=100)
    plt.show()

def clean_up_files(env_files, ants_files):
    for file in env_files + ants_files:
        try:
            os.remove(file)
        except OSError as e:
            print(f"Errore durante la rimozione del file {file}: {e}")

if __name__ == "__main__":
    env_files, ants_files = load_files()
    if len(env_files) == 0 or len(ants_files) == 0:
        print("Nessun file trovato per l'animazione.")
    else:
        create_animation(env_files, ants_files)
        clean_up_files(env_files, ants_files)

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.animation import FuncAnimation

# Function to load and validate the environment and ants files
def load_files():
    # Find all CSV files for environment and ants data
    env_files = sorted(glob.glob("cmake-build-debug-visual-studio/environment_*.csv"))
    ants_files = sorted(glob.glob("cmake-build-debug-visual-studio/ants_*.csv"))
    
    valid_env_files = []
    valid_ants_files = []

    # Check each pair of files for validity
    for env_file, ants_file in zip(env_files, ants_files):
        try:
            env_data = np.loadtxt(env_file, delimiter=",")
            ants_data = np.loadtxt(ants_file, delimiter=",", dtype=int)
            valid_env_files.append(env_file)
            valid_ants_files.append(ants_file)
        except Exception as e:
            print(f"Error in file {env_file} or {ants_file}: {e}")

    return valid_env_files, valid_ants_files

# Function to create an animation from the sequence of environment and ants files
def create_animation(env_files, ants_files):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Update function for each frame
    def update(frame):
        ax.clear()
        environment = np.loadtxt(env_files[frame], delimiter=",")
        ants = np.loadtxt(ants_files[frame], delimiter=",", dtype=int)

        ax.imshow(environment, cmap="Greys", origin="upper")
        ax.set_title(f"Environment and Ants - Frame {frame + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Plot each ant in the current frame
        for ant in ants:
            x, y, has_food = ant
            ax.scatter(x, y, c="red" if has_food else "blue", s=5)

    # Create and display the animation
    ani = FuncAnimation(fig, update, frames=len(env_files), interval=100)
    plt.show()

# Function to clean up (delete) files after animation is complete
def clean_up_files(env_files, ants_files):
    for file in env_files + ants_files:
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error while deleting file {file}: {e}")

if __name__ == "__main__":
    # Load environment and ants files
    env_files, ants_files = load_files()

    # Check if there are valid files to process
    if len(env_files) == 0 or len(ants_files) == 0:
        print("No files found for the animation.")
    else:
        # Create and display the animation
        create_animation(env_files, ants_files)

        # Clean up the files after use
        clean_up_files(env_files, ants_files)

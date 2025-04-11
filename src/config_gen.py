import os


def main():
    base_dir = "./configs"
    # Walk through all directories and files starting from base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file is a YAML file
            if file.endswith(".yaml"):
                # Construct the relative path to the file
                file_path = os.path.join(root, file)
                # Print the command in the desired format
                print(f'python main.py --config "{file_path}"')


if __name__ == "__main__":
    main()

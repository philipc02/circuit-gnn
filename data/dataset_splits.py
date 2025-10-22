import os
import random
import shutil

def create_splits(source_folder="../netlist_parser/graphs_star", output_root=".", split_ratio=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)

    # create folders for splits
    os.makedirs(output_root, exist_ok=True)
    train_folder = os.path.join(output_root, "train")
    val_folder = os.path.join(output_root, "val")
    test_folder = os.path.join(output_root, "test")

    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # list all graph files
    files = [f for f in os.listdir(source_folder) if f.endswith(".gpickle")]
    total = len(files)
    print(f"Found {total} graph files in {source_folder}.")

    # shuffle for randomness
    random.shuffle(files)

    # split files according to ratio
    n_train = round(total * split_ratio[0])
    n_val = round(total * split_ratio[1])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"Split sizes => Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # copy files into respective folders
    for subset_files, target_folder in [
        (train_files, train_folder),
        (val_files, val_folder),
        (test_files, test_folder)
    ]:
        for fname in subset_files:
            src_path = os.path.join(source_folder, fname)
            dst_path = os.path.join(target_folder, fname)
            shutil.copy2(src_path, dst_path)    # copy file along with metadata

    print("Data split complete!\n")
    print(f"Train: {train_folder}\nVal:   {val_folder}\nTest:  {test_folder}")

if __name__ == "__main__":
    create_splits()

import os
import random
import shutil
import numpy as np

def create_kfold_splits(source_folder="../netlist_parser/graphs_star_padded_homogeneous", output_root="data_kfold_padded_homogeneous", k=5, seed=42):
    random.seed(seed)

    os.makedirs(output_root, exist_ok=True)

    # list all graph files
    files = [f for f in os.listdir(source_folder) if f.endswith(".gpickle")]
    total = len(files)
    print(f"Found {total} graph files in {source_folder}.")

    # shuffle for randomness
    random.shuffle(files)

    # split files into k folds
    folds = np.array_split(np.array(files), k)

    for fold_idx in range(len(folds)):  # len(folds) should be k

        # create folders for splits
        fold_folder = os.path.join(output_root, f"fold_{fold_idx}")    # folder at the top
        train_folder = os.path.join(fold_folder, "train")
        val_folder = os.path.join(fold_folder, "val")
        test_folder = os.path.join(fold_folder, "test")
        for d in (train_folder, val_folder, test_folder):
            os.makedirs(d, exist_ok=True)

        test_files = folds[fold_idx]    # one fold is test file (because of changing fold_idx only exactly once)
        val_files = folds[(fold_idx + 1) % k] # next fold is validation file (because of changing fold_idx only exactly once, with wrap around for last fold)

        # rest are train files
        train_files = []
        for j in range(k):
            if j not in [fold_idx, (fold_idx + 1) % k]: 
                train_files.extend(folds[j])

        print(f"Fold {fold_idx} => Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}\n")

        # copy files into respective folders
        for subset_files, target_folder in [
            (train_files, train_folder),
            (val_files, val_folder),
            (test_files, test_folder),
        ]:
            for fname in subset_files:
                shutil.copy2(os.path.join(source_folder, fname), os.path.join(target_folder, fname))   # copy file along with metadata

    print("k-fold dataset generated.\n")


if __name__ == "__main__":
    create_kfold_splits()

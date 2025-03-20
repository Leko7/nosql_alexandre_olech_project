from UCRDataset import UCRDataset

import time
from torch.utils.data import DataLoader
import torch

def evaluation(selected_dist, selected_norm, datasets_list, root_dir, batch_size, distance_metrics, normalizations):
    distance = distance_metrics[selected_dist]
    normalization = normalizations[selected_norm]
    # add selected_dist and distances dict instead, same with norms
    avg_acc = 0
    start_time = time.time()

    for i in range(len(datasets_list)):
        data_dir = datasets_list[i]

        # Create dataset instances
        train_dataset = UCRDataset(root_dir, data_dir, split="train", transform=normalization)
        test_dataset = UCRDataset(root_dir, data_dir, split="test", transform=normalization)

        # Create DataLoader

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        # Compute 1-NN classification
        for x_test, y_test in test_loader:
            best_distances = torch.full((x_test.shape[0],), torch.inf, dtype=torch.float32)
            class_preds = torch.zeros((x_test.shape[0],), dtype=torch.int64)

            for x_train, y_train in train_loader:
                # Compute pairwise distances
                distances = distance(x_test, x_train) # shape (test_batch_size, train_batch_size)

                # Get the nearest neighbor index and its corresponding distance
                min_distances, nearest_indices = torch.min(distances, dim=1)

                # Update if the new min distance is smaller
                mask = min_distances < best_distances
                best_distances[mask] = min_distances[mask]
                class_preds[mask] = y_train[nearest_indices[mask]]

            # Compute accuracy
            correct += (class_preds == y_test).sum().item()
            total += y_test.shape[0]

        # Final accuracy
        acc = correct / total
        avg_acc += acc
        print(f"Dataset: {data_dir}, Accuracy on the test set: {acc * 100:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - start_time

    avg_acc /= len(datasets_list)
    n_minutes = elapsed_time // 60
    n_secs = elapsed_time%60

    print(f"Execution time: {int(n_minutes)} min {round(n_secs,2)} secs.\n")

    return {
        "Distance Measure": selected_dist,
        "Scaling Method": selected_norm,
        "Average Accuracy": avg_acc * 100
        }
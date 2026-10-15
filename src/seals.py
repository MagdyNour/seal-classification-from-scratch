
import numpy as np
import os
import matplotlib.pyplot as plt


# Locate the project and data directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, "data")


# Load seal data
def load_seal_data():
    """Load the seal dataset from the project's data directory."""
    try:
        # TRAIN
        train_images = np.loadtxt(
            os.path.join(data_dir, "seals_images_train.csv")
        )
        train_labels_features = np.loadtxt(
            os.path.join(data_dir, "seals_train.csv")
        )
        train_features = train_labels_features[:, 1:]
        train_labels = train_labels_features[:, 0].astype(int)

        # TEST
        test_images = np.loadtxt(
            os.path.join(data_dir, "seals_images_test.csv")
        )
        test_labels_features = np.loadtxt(
            os.path.join(data_dir, "seals_test.csv")
        )
        test_features = test_labels_features[:, 1:]
        test_labels = test_labels_features[:, 0].astype(int)

        print(f"Training data shape: {train_features.shape}")
        print(f"Test data shape: {test_features.shape}")
        print(f"Class distribution - Train: {np.bincount(train_labels)}")
        print(f"Class distribution - Test: {np.bincount(test_labels)}")

        return (
            train_images,
            train_features,
            train_labels,
            test_images,
            test_features,
            test_labels,
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading files: {e}")
        return None, None, None, None, None, None


# Question f
def plot_seal_image(index, images_array, labels_array):
    """Plot the seal image corresponding to the given index."""
    image = images_array[index].reshape(64, 64)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.title(f"Seal Image #{index}, Label: {labels_array[index]}")
    plt.axis("off")
    plt.show()

    return image


# Question g: Find the 3 closest images to image #58
def plot_closest_seals(train_images, train_features, train_labels):
    distances = np.sqrt(
        np.sum((train_features - train_features[58]) ** 2, axis=1)
    )
    closest_indices = np.argsort(distances)[1:4]

    plt.figure(figsize=(12, 3))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(train_images[58].reshape(64, 64), cmap="gray")
    plt.title(f"#58\nLabel: {train_labels[58]}")
    plt.axis("off")

    # Three closest images
    for i, idx in enumerate(closest_indices):
        plt.subplot(1, 4, i + 2)
        plt.imshow(train_images[idx].reshape(64, 64), cmap="gray")
        plt.title(f"#{idx}\nLabel: {train_labels[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Image #58 label: {train_labels[58]}")
    print(f"Closest labels: {train_labels[closest_indices]}")
    print(
        f"All same? "
        f"{all(train_labels[closest_indices] == train_labels[58])}"
    )


# Decision Tree Implementation

# From exercises: binaryEntropy, findBestSplit and findBestFeatureToSplit
def binaryEntropy(labels):
    """Compute entropy of binary class labels."""
    if len(labels) == 0:
        return 0

    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def findBestSplit(values, labels):
    """Find the best split point for a single feature."""
    if len(values) == 0:
        return None, float("inf")

    currentEntropy = binaryEntropy(labels)
    best_gain = -1
    best_threshold = None

    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]

    for i in range(1, len(sorted_values)):
        if sorted_values[i] != sorted_values[i - 1]:
            threshold = (sorted_values[i] + sorted_values[i - 1]) / 2

            left_part = values <= threshold
            right_part = values > threshold

            if np.sum(left_part) > 0 and np.sum(right_part) > 0:
                left_entropy = binaryEntropy(labels[left_part])
                right_entropy = binaryEntropy(labels[right_part])

                newEntropy = (
                    np.sum(left_part) * left_entropy
                    + np.sum(right_part) * right_entropy
                ) / len(labels)

                info_gain = currentEntropy - newEntropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_threshold = threshold

    return best_threshold, best_gain


def findBestFeatureToSplit(dataset, labels):
    """Find the best feature to split on in the dataset."""
    bestFeature = None
    bestSplit = None
    best_gain = -1

    num_features = dataset.shape[1]

    for feature in range(num_features):
        split, gain = findBestSplit(dataset[:, feature], labels)

        if gain > best_gain:
            best_gain = gain
            bestFeature = feature
            bestSplit = split

    return bestFeature, bestSplit, best_gain


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _most_common_label(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    def _build_tree(self, dataset, labels, depth=0):
        num_samples, num_features = dataset.shape

        # Stopping criteria
        if (
            depth >= self.max_depth
            or num_samples < self.min_samples_split
            or len(np.unique(labels)) == 1
        ):
            return self._most_common_label(labels)

        # Find the best feature and split
        feature, split, gain = findBestFeatureToSplit(dataset, labels)

        # If no valid split is found, make a leaf
        if feature is None or split is None or gain <= 0:
            return self._most_common_label(labels)

        # Split the dataset
        left_indices = dataset[:, feature] <= split
        right_indices = dataset[:, feature] > split

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return self._most_common_label(labels)

        # Build subtrees recursively
        left_subtree = self._build_tree(
            dataset[left_indices], labels[left_indices], depth + 1
        )
        right_subtree = self._build_tree(
            dataset[right_indices], labels[right_indices], depth + 1
        )

        return (feature, split, left_subtree, right_subtree)

    def fit(self, dataset, labels):
        self.root = self._build_tree(dataset, labels)

    def predict(self, dataset):
        return np.array([
            self._traverse_tree(x, self.root) for x in dataset
        ])

    def _traverse_tree(self, x, node):
        if not isinstance(node, tuple):
            return node

        feature, split, left_subtree, right_subtree = node

        if x[feature] <= split:
            return self._traverse_tree(x, left_subtree)

        return self._traverse_tree(x, right_subtree)


# For question 1a
def confusion_matrix(y_true, y_pred, classes=[0, 1]):
    """Compute confusion matrix."""
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    return cm


def print_metrics(y_true, y_pred):
    """Print confusion matrix and accuracy."""
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print("Actual\\Predicted   Harp(0)  Hooded(1)")
    print(f"Harp (0)          {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"Hooded (1)        {cm[1, 0]:6d}  {cm[1, 1]:6d}")

    accuracy = np.mean(y_pred == y_true)
    print(f"\nAccuracy: {accuracy:.4f}")

    return cm, accuracy


# Problem 1b: Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Tune decision tree hyperparameters."""
    print("PROBLEM 1b: HYPERPARAMETER TUNING")
    print("=" * 50)

    depths = [3, 5, 8, 10, 12, 15]

    best_accuracy = 0
    best_depth = 0
    best_cm = None

    print("Testing Different Max Depths:")
    print("-" * 30)

    for depth in depths:
        dt = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=2
        )
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        cm, accuracy = print_metrics(y_test, y_pred)

        print(f"Depth {depth:2d}: Accuracy = {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
            best_cm = cm

    return best_depth, best_accuracy, best_cm


# Problem 1c: kNN Implementation
# AI assistance was used to correct errors in the original kNN loop.
def simple_knn_predict(X_train, y_train, X_test, k=3):
    """kNN prediction with raw features."""
    predictions = []

    for test_point in X_test:
        distances = np.sqrt(
            np.sum((X_train - test_point) ** 2, axis=1)
        )
        nearest_indices = np.argsort(distances)[:k]
        neighbor_labels = y_train[nearest_indices]

        # Count votes for each class, with tie-breaking
        counts = np.bincount(neighbor_labels, minlength=2)

        if counts[0] == counts[1]:
            predictions.append(neighbor_labels[0])
        else:
            predictions.append(np.argmax(counts))

    return np.array(predictions)


def test_simple_knn(
    X_train, y_train, X_test, y_test,
    k_values=[1, 3, 5, 15, 25]
):
    """Test kNN with different k values."""
    print("\nPROBLEM 1c: kNN CLASSIFIER PERFORMANCE")
    print("k  Accuracy  Sensitivity  Specificity  F1-Score")
    print("-" * 50)

    results = {}

    for k in k_values:
        y_pred = simple_knn_predict(X_train, y_train, X_test, k)
        cm = confusion_matrix(y_test, y_pred)

        accuracy = np.mean(y_pred == y_test)
        sensitivity = (
            cm[1, 1] / (cm[1, 1] + cm[1, 0])
            if (cm[1, 1] + cm[1, 0]) > 0 else 0
        )
        specificity = (
            cm[0, 0] / (cm[0, 0] + cm[0, 1])
            if (cm[0, 0] + cm[0, 1]) > 0 else 0
        )
        precision = (
            cm[1, 1] / (cm[1, 1] + cm[0, 1])
            if (cm[1, 1] + cm[0, 1]) > 0 else 0
        )
        f1 = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0 else 0
        )

        print(
            f"{k:2d}  {accuracy:.4f}    {sensitivity:.4f}      "
            f"{specificity:.4f}      {f1:.4f}"
        )

        results[k] = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "cm": cm,
        }

    return results


# Problem 1h: Misclassification Analysis
def analyze_misclassifications(
    X_train, y_train, X_test, y_test, test_images, k=3
):
    """Analyze misclassified images."""
    print("\nPROBLEM 1h: MISCLASSIFICATION ANALYSIS")
    print("=" * 50)

    knn_predictions = simple_knn_predict(
        X_train, y_train, X_test, k
    )
    misclassified = np.where(knn_predictions != y_test)[0]

    print(
        f"Found {len(misclassified)} misclassified images "
        f"out of {len(y_test)} total"
    )
    print(f"Accuracy: {(1 - len(misclassified) / len(y_test)):.4f}")

    if len(misclassified) > 0:
        plt.figure(figsize=(15, 6))

        for i, idx in enumerate(misclassified[:5]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(test_images[idx].reshape(64, 64), cmap="gray")
            plt.title(
                f"True: {y_test[idx]}, Pred: {knn_predictions[idx]}"
            )
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        print("\nFirst 5 misclassified examples:")
        for idx in misclassified[:5]:
            print(
                f"Image #{idx}: True label = {y_test[idx]}, "
                f"Predicted = {knn_predictions[idx]}"
            )

    return misclassified, knn_predictions


# Main execution
if __name__ == "__main__":
    (
        train_images,
        X_train,
        y_train,
        test_images,
        X_test,
        y_test,
    ) = load_seal_data()

    if X_train is not None:
        # Display a sample image
        sample_image = train_images[0].reshape(64, 64)
        plt.figure()
        plt.title(f"Sample Seal Image, Label: {y_train[0]}")
        plt.imshow(sample_image)
        plt.show()

        # Plot candidate image #58
        print("Showing seal image #58:")
        plot_seal_image(58, train_images, y_train)

        # Question g
        plot_closest_seals(train_images, X_train, y_train)

        # Problem 1a: Decision Tree
        print("\n" + "=" * 70)
        print("PROBLEM 1a: DECISION TREE CLASSIFIER")
        print("=" * 70)

        dt = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5
        )
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        cm_dt, accuracy_dt = print_metrics(y_test, y_pred_dt)

        # Problem 1b: Hyperparameter Tuning
        print("\n" + "=" * 70)
        best_depth, best_accuracy, best_cm = hyperparameter_tuning(
            X_train, y_train, X_test, y_test
        )

        print("\n" + "=" * 70)
        print("TUNING RESULTS SUMMARY:")
        print(f"Original Accuracy (depth=8): {accuracy_dt:.4f}")
        print(
            f"Best Tuned Accuracy (depth={best_depth}): "
            f"{best_accuracy:.4f}"
        )
        improvement = best_accuracy - accuracy_dt
        print(
            f"Improvement: {improvement:+.4f} "
            f"({improvement * 100:+.2f}%)"
        )

        # Problem 1c: kNN with raw features
        print("\n" + "=" * 70)
        print("PROBLEM 1c: kNN CLASSIFIER PERFORMANCE")
        print("=" * 70)

        knn_results = test_simple_knn(
            X_train, y_train, X_test, y_test
        )

        # Problem 1h: Misclassification Analysis
        print("\n" + "=" * 70)
        print("PROBLEM 1h: MISCLASSIFICATION ANALYSIS")
        print("=" * 70)

        misclassified, knn_predictions = analyze_misclassifications(
            X_train, y_train, X_test, y_test, test_images, k=3
        )

        # Final comparison
        print("\n" + "=" * 70)
        print("FINAL PERFORMANCE COMPARISON:")
        print("=" * 70)
        print(f"Decision Tree (tuned): {best_accuracy:.4f}")
        print(f"kNN (k=3): {knn_results[3]['accuracy']:.4f}")
        print(f"kNN (best k=15): {knn_results[15]['accuracy']:.4f}")

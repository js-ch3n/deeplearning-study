import argparse
import os
import random

import numpy as np
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def save_test_samples(data_folder="data", output_dir="test_samples",
                      samples_per_class=1, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    raw_transform = transforms.ToTensor()
    test_dataset = datasets.CIFAR10(
        root=data_folder,
        train=False,
        download=True,
        transform=raw_transform,
   )

    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_dataset):
        class_indices[label].append(idx)

    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for cls_idx in range(10):
        indices = random.sample(class_indices[cls_idx],
                                min(samples_per_class, len(class_indices[cls_idx])))
        for i, idx in enumerate(indices):
            image_tensor, label = test_dataset[idx]
            image = transforms.ToPILImage()(image_tensor)
            filename = f"{cls_idx:02d}_{CIFAR10_CLASSES[cls_idx]}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved.append((filename, CIFAR10_CLASSES[label]))
            print(f"  saved: {filename}  (label: {CIFAR10_CLASSES[label]})")

    print(f"\nTotal {len(saved)} images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Save samples from CIFAR-10 test set"
    )
    parser.add_argument("--data", type=str, default="data",
                        help="Path to data folder (default: data)")
    parser.add_argument("--output", type=str, default="test_samples",
                        help="Output directory (default: test_samples)")
    parser.add_argument("--per-class", type=int, default=1,
                        help="Number of samples per class (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    save_test_samples(
        data_folder=args.data,
        output_dir=args.output,
        samples_per_class=args.per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

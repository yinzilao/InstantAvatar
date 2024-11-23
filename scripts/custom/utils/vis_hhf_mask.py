import numpy as np
import matplotlib.pyplot as plt

def visualize_hhf_mask(mask_path, save_path=None):
    # Load the mask
    mask = np.load(mask_path)
    
    # Assert unique values
    unique_values = np.unique(mask)
    expected_values = np.array([0.0, 0.5, 1.0])
    assert np.allclose(np.sort(unique_values), expected_values), \
        f"Expected values {expected_values}, but got {unique_values}"
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('HHF Weight Mask')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    import os
    import glob

    mask_dir = "data/custom/e1/hhf_weight_masks"
    npy_files = glob.glob(os.path.join(mask_dir, "*.npy"))

    for mask_path in sorted(npy_files)[::100]: # step size of 100
        save_path = mask_path.replace('.npy', '_vis.png')
        print(f"Processing {os.path.basename(mask_path)}")
        visualize_hhf_mask(mask_path, save_path) 
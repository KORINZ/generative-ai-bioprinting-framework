import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os
import seaborn as sns
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from beta_cvae_hyperparameters import LATENT_DIM, CONDITION_DIM, BETA


class GridImageDataset(Dataset):
    def __init__(
        self,
        grid_index_csv,
        cross_power_csv,
        ink_ph_csv,
        image_dir,
        transform=None,
        augment=True,
        scaler=None,
        fit_scaler=True,
        base_data=None,
        image_subset=None,
        verbose=True,
        apply_intensity_augmentation=True,
    ) -> None:
        """Custom dataset that loads grid images, matches them with rheology and pH data, and applies augmentations"""
        self.grid_df = pd.read_csv(grid_index_csv)
        self.rheology_df = pd.read_csv(cross_power_csv)
        self.ph_df = pd.read_csv(ink_ph_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.augment = augment
        self.apply_intensity_augmentation = apply_intensity_augmentation

        # If base_data is provided, use it directly
        if base_data is not None:
            self.base_data = base_data
        else:
            # Filter for grid shape only
            self.grid_df = self.grid_df[self.grid_df["shape"] == "grid"]

            # If image_subset is provided, filter to only those images
            if image_subset is not None:
                subset_image_ids = {img["image_id"] for img in image_subset}
                self.grid_df = self.grid_df[
                    self.grid_df["image_id"].isin(subset_image_ids)
                ]

            # Process data
            self.base_data = []
            for _, row in self.grid_df.iterrows():
                image_id = row["image_id"]
                ink_type_ph = row["ink_type_ph"]
                intensity = row["intensity"]

                # Check contain_sps column
                contain_sps = row.get("contain_sps", "")

                # Parse ink type
                ink_info = self.parse_ink_type(ink_type_ph)
                if ink_info is None:
                    continue

                # Find matching rheology data
                rheology_match = self.find_rheology_match(ink_info, self.rheology_df)
                if rheology_match is None:
                    continue

                # Format image_id with leading zeros (assuming 4 digits)
                image_id_str = str(image_id).zfill(4)

                # Check if image exists
                image_path = self.image_dir / f"resized_IMG_{image_id_str}_masked.png"
                if not image_path.exists():
                    print(f"Warning: Image not found at {image_path}")
                    continue

                # Handle contain_sps == "no" case
                if contain_sps == "no":
                    if self.apply_intensity_augmentation:
                        # Create entries for multiple intensities for intensity augmentation
                        intensities = [0, 100]
                        for int_val in intensities:
                            self.base_data.append(
                                {
                                    "image_path": image_path,
                                    "image_id": image_id,
                                    "intensity": int_val / 100.0,
                                    "eta0": rheology_match["eta0"],
                                    "m": rheology_match["m"],
                                    "n": rheology_match["n"],
                                    "ph_content": 0.0,
                                    "tan_delta": rheology_match["tan_delta"],
                                    "original_intensity": intensity / 100.0,
                                }
                            )
                    else:
                        # if not intensity augmentation: use the original intensity, but still set ph_content to 0
                        self.base_data.append(
                            {
                                "image_path": image_path,
                                "image_id": image_id,
                                "intensity": intensity / 100.0,
                                "eta0": rheology_match["eta0"],
                                "m": rheology_match["m"],
                                "n": rheology_match["n"],
                                "ph_content": 0.0,
                                "tan_delta": rheology_match["tan_delta"],
                                "original_intensity": intensity / 100.0,
                            }
                        )
                else:
                    # Normal case - get Ph content
                    ph_content = self.get_ph_content(ink_type_ph)
                    if ph_content is None:
                        print(
                            f"Warning: No Ph content found for {ink_type_ph}; image_id {image_id}"
                        )
                        continue

                    # Add data entry
                    self.base_data.append(
                        {
                            "image_path": image_path,
                            "image_id": image_id,
                            "intensity": intensity / 100.0,
                            "eta0": rheology_match["eta0"],
                            "m": rheology_match["m"],
                            "n": rheology_match["n"],
                            "ph_content": ph_content,
                            "tan_delta": rheology_match["tan_delta"],
                            "original_intensity": intensity / 100.0,
                        }
                    )

            if verbose:
                print(f"Found {len(self.base_data)} valid image-rheology-pH pairs")
                unique_images = len(set(d["image_id"] for d in self.base_data))
                print(f"  From {unique_images} unique images")

        # Set up StandardScaler
        if len(self.base_data) > 0:
            conditions_array = np.array(
                [
                    [
                        d["intensity"],
                        np.log10(d["eta0"]),
                        np.log10(d["m"]),
                        d["n"],
                        np.log1p(d["ph_content"]),
                        d["tan_delta"],
                    ]
                    for d in self.base_data
                ]
            )

            if scaler is None and fit_scaler:
                self.scaler = StandardScaler()
                self.scaler.fit(conditions_array)
                if verbose:
                    print("Fitted StandardScaler with statistics:")
                    print(f"  Mean: {self.scaler.mean_}")
                    print(f"  Std:  {self.scaler.scale_}")
            else:
                self.scaler = scaler
                if scaler is not None and verbose:
                    print("Using provided StandardScaler")
        else:
            self.scaler = None

        # Store augmented data
        self._rebuild_augmented_data()

    def _rebuild_augmented_data(self) -> None:
        """Rebuild augmented data from base_data"""
        self.data_with_augmentation = []
        if self.augment:
            for data in self.base_data:
                for rotation in [0, 90, 180, 270]:
                    for flip in [False, True]:
                        aug_data = data.copy()
                        aug_data["rotation"] = rotation
                        aug_data["h_flip"] = flip
                        self.data_with_augmentation.append(aug_data)
        else:
            # No augmentation - just copy the base data
            self.data_with_augmentation = [d.copy() for d in self.base_data]
            for d in self.data_with_augmentation:
                d["rotation"] = 0
                d["h_flip"] = False

    def parse_ink_type(self, ink_type_ph) -> dict | None:
        """Parse ink type to extract components and concentrations"""
        # Pattern for ALG_HA composite: ALG2.0HA0.5
        composite_pattern = r"ALG(\d+\.?\d*)HA(\d+\.?\d*)"
        # Pattern for single component: ALG2.0 or HA1.0
        single_pattern = r"(ALG|HA)(\d+\.?\d*)"

        composite_match = re.match(composite_pattern, ink_type_ph)
        if composite_match:
            alg_conc = float(composite_match.group(1))
            ha_conc = float(composite_match.group(2))
            return {"type": "composite", "alg_conc": alg_conc, "ha_conc": ha_conc}

        single_match = re.match(single_pattern, ink_type_ph)
        if single_match:
            ink_type = single_match.group(1)
            concentration = float(single_match.group(2))
            return {
                "type": "single",
                "ink_type": ink_type,
                "concentration": concentration,
            }

        return None

    def find_rheology_match(self, ink_info, rheology_df) -> pd.Series | None:
        """Find matching rheology data based on ink info"""
        if ink_info["type"] == "single":
            if ink_info["ink_type"] == "ALG":
                # Match ALG inks that start with ALG-I1G-Ph_X.X_ and don't contain HA
                pattern = f"^ALG-I1G-Ph_{ink_info['concentration']}_"
                matches = rheology_df[
                    (rheology_df["ink_type"].str.match(pattern))
                    & (~rheology_df["ink_type"].str.contains("HA-HQ-Ph"))
                ]
            else:  # HA
                # Match HA inks that start with HA-HQ-Ph_X.X_
                pattern = f"^HA-HQ-Ph_{ink_info['concentration']}_"
                matches = rheology_df[rheology_df["ink_type"].str.match(pattern)]
        else:  # composite
            # Match composite inks: ALG-I1G-Ph_X.X_HA-HQ-Ph_Y.Y_
            pattern = (
                f"ALG-I1G-Ph_{ink_info['alg_conc']}_HA-HQ-Ph_{ink_info['ha_conc']}_"
            )
            matches = rheology_df[
                rheology_df["ink_type"].str.contains(pattern, regex=False)
            ]

        if len(matches) > 0:
            return matches.iloc[0]
        return None

    def get_ph_content(self, ink_type_ph) -> float | None:
        """Get Ph content for the ink type"""
        ph_match = self.ph_df[self.ph_df["ink_type_ph"] == ink_type_ph]
        if len(ph_match) > 0:
            return ph_match.iloc[0]["mmol_ph_ml"]
        return None

    def get_base_indices(self) -> list[int]:
        """Return indices that correspond to base images (no augmentation)"""
        if self.augment:
            return list(
                range(0, len(self.data_with_augmentation), 8)
            )  # Changed from 4 to 8
        else:
            return list(range(len(self.data_with_augmentation)))

    def __len__(self) -> int:
        """Return total number of samples (including augmentations)"""
        return len(self.data_with_augmentation)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image and condition for the given index, applying augmentations as needed"""
        data = self.data_with_augmentation[idx]

        # Load image as RGB instead of grayscale
        image = Image.open(data["image_path"]).convert("RGB")

        # Apply rotation
        if data["rotation"] > 0:
            image = image.rotate(data["rotation"])

        # Apply horizontal flip
        if data["h_flip"]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize to manageable size (e.g., 128x128 for training)
        image = image.resize((128, 128), Image.LANCZOS)

        # Convert to tensor
        image = transforms.ToTensor()(image)

        # Create condition vector
        condition = np.array(
            [
                data["intensity"],
                np.log10(data["eta0"]),
                np.log10(data["m"]),
                data["n"],
                np.log1p(data["ph_content"]),
                data["tan_delta"],
            ]
        )

        # Apply scaling if scaler exists
        if hasattr(self, "scaler") and self.scaler is not None:
            condition = self.scaler.transform(condition.reshape(1, -1))[0]

        condition = torch.tensor(condition, dtype=torch.float32)

        return image, condition


def create_train_val_test_datasets(
    grid_index_csv,
    cross_power_csv,
    ink_ph_csv,
    image_dir,
    train_ratio=0.75,
    val_ratio=0.15,
    test_ratio=0.10,
    random_state=42,
) -> tuple[GridImageDataset, GridImageDataset, GridImageDataset]:
    """Create train, validation, and test datasets with proper data split"""
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1"

    # Create a dataset without augmentation to get the base data
    print("Loading and processing data...")

    # Handle the split differently to avoid data leakage
    grid_df = pd.read_csv(grid_index_csv)
    grid_df = grid_df[grid_df["shape"] == "grid"]

    # Create a list of unique images first (before intensity augmentation)
    unique_images = []
    for _, row in grid_df.iterrows():
        unique_images.append(
            {
                "image_id": row["image_id"],
                "ink_type_ph": row["ink_type_ph"],
                "intensity": row["intensity"],
                "contain_sps": row.get("contain_sps", ""),
            }
        )

    # Split at the image level
    n_images = len(unique_images)
    indices = list(range(n_images))

    # First split to separate test set
    temp_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_state
    )

    # Split remaining into train and validation
    # Calculate the adjusted validation ratio for the remaining data
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        temp_indices, test_size=adjusted_val_ratio, random_state=random_state
    )

    # Create train, validation, and test image lists
    train_images = [unique_images[i] for i in train_indices]
    val_images = [unique_images[i] for i in val_indices]
    test_images = [unique_images[i] for i in test_indices]

    print(
        f"Split {n_images} unique images: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test"
    )

    # Create the datasets with the split data
    train_dataset = GridImageDataset(
        grid_index_csv=grid_index_csv,
        cross_power_csv=cross_power_csv,
        ink_ph_csv=ink_ph_csv,
        image_dir=image_dir,
        augment=True,
        apply_intensity_augmentation=True,  # Enable intensity augmentation
        fit_scaler=True,
        image_subset=train_images,  # Pass the train images
        verbose=True,
    )

    val_dataset = GridImageDataset(
        grid_index_csv=grid_index_csv,
        cross_power_csv=cross_power_csv,
        ink_ph_csv=ink_ph_csv,
        image_dir=image_dir,
        augment=False,
        apply_intensity_augmentation=False,  # No intensity augmentation for val
        scaler=train_dataset.scaler,  # Use scaler fitted on training data
        fit_scaler=False,
        image_subset=val_images,  # Pass the val images
        verbose=True,
    )

    test_dataset = GridImageDataset(
        grid_index_csv=grid_index_csv,
        cross_power_csv=cross_power_csv,
        ink_ph_csv=ink_ph_csv,
        image_dir=image_dir,
        augment=False,
        apply_intensity_augmentation=False,  # No intensity augmentation for test
        scaler=train_dataset.scaler,  # Use scaler fitted on training data
        fit_scaler=False,
        image_subset=test_images,  # Pass the test images
        verbose=True,
    )

    print("\nData split summary:")
    print(f"Total unique images: {n_images}")
    print(f"Train images: {len(train_images)} ({len(train_images)/n_images*100:.1f}%)")
    print(f"Val images: {len(val_images)} ({len(val_images)/n_images*100:.1f}%)")
    print(f"Test images: {len(test_images)} ({len(test_images)/n_images*100:.1f}%)")
    print(
        f"Train total samples (with intensity/rotation augmentation): {len(train_dataset)}"
    )
    print(f"Val total samples: {len(val_dataset)}")
    print(f"Test total samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


class Encoder(nn.Module):
    def __init__(self, latent_dim, condition_dim) -> None:
        """Encoder network with four convolutional layers and condition concatenation"""

        super().__init__()
        self.condition_dim = condition_dim

        # Four convolutional layers with stride 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 16 -> 8

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Flatten size: 256 * 8 * 8 = 16384
        self.fc_condition = nn.Linear(condition_dim, 128)
        self.fc_mu = nn.Linear(16384 + 128, latent_dim)
        self.fc_logvar = nn.Linear(16384 + 128, latent_dim)

    def forward(self, x, c) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image x and condition c into latent mean and log variance"""

        # Convolutional encoding with four layers
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        # Flatten
        h = h.view(h.size(0), -1)

        # Condition encoding
        c_encoded = F.relu(self.fc_condition(c))

        # Concatenate image features with condition
        h = torch.cat([h, c_encoded], dim=1)

        # Output mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim) -> None:
        """Decoder network that takes latent vector and condition to reconstruct the image"""

        super().__init__()
        self.condition_dim = condition_dim

        # Project latent + condition to feature map
        self.fc_condition = nn.Linear(condition_dim, 128)
        self.fc = nn.Linear(latent_dim + 128, 256 * 8 * 8)

        # Four convolutional layers (used after bilinear upsampling)
        self.conv1 = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1
        )  # No size change
        self.conv2 = nn.Conv2d(
            128, 64, kernel_size=3, stride=1, padding=1
        )  # No size change
        self.conv3 = nn.Conv2d(
            64, 32, kernel_size=3, stride=1, padding=1
        )  # No size change
        self.conv4 = nn.Conv2d(
            32, 3, kernel_size=3, stride=1, padding=1
        )  # Output 3 channels for RGB

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, z, c) -> torch.Tensor:
        """Decode latent vector z and condition c to reconstruct the image"""

        # Condition encoding
        c_encoded = F.relu(self.fc_condition(c))

        # Concatenate latent with condition
        h = torch.cat([z, c_encoded], dim=1)
        h = F.relu(self.fc(h))

        # Reshape to 8x8 feature map
        h = h.view(h.size(0), 256, 8, 8)

        # Upsample and apply convolutions
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)  # 16
        h = F.relu(self.bn1(self.conv1(h)))

        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)  # 32
        h = F.relu(self.bn2(self.conv2(h)))

        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)  # 64
        h = F.relu(self.bn3(self.conv3(h)))

        h = F.interpolate(
            h, scale_factor=2, mode="bilinear", align_corners=False
        )  # 128
        x_recon = torch.sigmoid(self.conv4(h))

        return x_recon


class BetaCVAE(nn.Module):
    def __init__(self, latent_dim=32, condition_dim=6, beta=5.0) -> None:
        """Conditional Variational Autoencoder with beta weighting for KL divergence"""

        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, condition_dim)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)"""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input image x and condition c, sample from latent space, and decode to reconstruct image"""

        # Encode
        mu, logvar = self.encoder(x, c)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z, c)

        return x_recon, mu, logvar

    def loss_function(
        self, x_recon, x, mu, logvar
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the total loss as a combination of reconstruction loss and KL divergence, weighted by beta"""

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


def train_epoch(model, dataloader, optimizer, device) -> tuple[float, float, float]:
    """Train the model for one epoch and return average total loss, reconstruction loss, and KL loss"""

    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0

    for _, (images, conditions) in enumerate(dataloader):
        images = images.to(device)
        conditions = conditions.to(device)

        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, logvar = model(images, conditions)

        # Calculate loss
        loss, recon_loss, kl_loss = model.loss_function(x_recon, images, mu, logvar)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track losses
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()

    # Average losses
    avg_loss = epoch_loss / len(dataloader)
    avg_recon_loss = epoch_recon_loss / len(dataloader)
    avg_kl_loss = epoch_kl_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_loss


def validate_epoch(model, dataloader, device) -> tuple[float, float, float]:
    """Validate the model for one epoch and return average total loss, reconstruction loss, and KL loss"""

    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0

    with torch.no_grad():
        for batch_idx, (images, conditions) in enumerate(dataloader):
            images = images.to(device)
            conditions = conditions.to(device)

            # Forward pass
            x_recon, mu, logvar = model(images, conditions)

            # Calculate loss
            loss, recon_loss, kl_loss = model.loss_function(x_recon, images, mu, logvar)

            # Track losses
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()

    # Average losses
    avg_loss = val_loss / len(dataloader)
    avg_recon_loss = val_recon_loss / len(dataloader)
    avg_kl_loss = val_kl_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_loss


def train_model(
    train_dataset,
    val_dataset,
    output_dir,
    epochs=200,
    lr=1e-3,
    batch_size=64,
    patience=10,
) -> dict:
    """Train the BetaCVAE model and return training history"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = BetaCVAE(latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM, beta=BETA).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    # Train for specified epochs
    for epoch in range(epochs):
        # Training
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device
        )
        train_losses.append((train_loss, train_recon, train_kl))

        # Validation
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device)
        val_losses.append((val_loss, val_recon, val_kl))

        # Print progress
        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (R: {train_recon:.4f}, KL: {train_kl:.4f}) | "
            f"Val Loss: {val_loss:.4f} (R: {val_recon:.4f}, KL: {val_kl:.4f})"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{output_dir}/beta_cvae_best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load the best model
    model.load_state_dict(
        torch.load(f"{output_dir}/beta_cvae_best_model.pth", weights_only=True)
    )

    # Plot losses and save reconstructions
    plot_losses(train_losses, val_losses, output_dir, best_epoch)
    save_reconstructions(model, val_loader, device, output_dir)

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Return the training history
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def plot_losses(train_losses, val_losses, output_dir, best_epoch) -> None:
    """Plot training and validation losses over epochs and save the figure"""

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    legend_props = dict(
        framealpha=1, fancybox=False, edgecolor="black", loc="upper right"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    ax1.plot(train_losses[:, 0], label="Train")
    ax1.plot(val_losses[:, 0], label="Validation")
    ax1.axvline(
        x=best_epoch - 1,
        color="gray",
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.legend(**legend_props)
    ax1.set_yscale("log")
    ax1.grid(True)

    # Reconstruction loss
    ax2.plot(train_losses[:, 1], label="Train")
    ax2.plot(val_losses[:, 1], label="Validation")
    ax2.axvline(
        x=best_epoch - 1,
        color="gray",
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )
    ax2.set_title("Reconstruction Loss")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Loss (MSE)", fontweight="bold")
    ax2.legend(**legend_props)
    ax2.set_yscale("log")
    ax2.grid(True)

    # KL loss
    ax3.plot(train_losses[:, 2], label="Train")
    ax3.plot(val_losses[:, 2], label="Validation")
    ax3.axvline(
        x=best_epoch - 1,
        color="gray",
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )
    ax3.set_title("KL Loss")
    ax3.set_xlabel("Epoch", fontweight="bold")
    ax3.set_ylabel("Loss (KLD)", fontweight="bold")
    ax3.legend(**legend_props)
    ax3.set_yscale("log")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_losses.png", bbox_inches="tight", dpi=600)
    plt.close()


def save_reconstructions(model, val_loader, device, output_dir) -> None:
    """Save original and reconstructed images from the validation set"""

    # Set model to evaluation mode
    model.eval()

    # Get one batch from validation
    images, conditions = next(iter(val_loader))
    images = images[:8].to(device)
    conditions = conditions[:8].to(device)

    with torch.no_grad():
        x_recon, _, _ = model(images, conditions)

    # Create subfolder for individual figures
    individual_dir = Path(output_dir) / "individual_reconstructions"
    individual_dir.mkdir(exist_ok=True)

    # Save individual figures at exactly 800x800 pixels
    for i in range(8):
        # Original
        orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(orig_img)
        ax.axis("off")
        ax.set_position([0, 0, 1, 1])
        plt.savefig(individual_dir / f"original_{i+1:02d}.png", dpi=100, pad_inches=0)
        plt.close()

        # Reconstructed
        recon_img = x_recon[i].cpu().numpy().transpose(1, 2, 0)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(recon_img)
        ax.axis("off")
        ax.set_position([0, 0, 1, 1])
        plt.savefig(
            individual_dir / f"reconstructed_{i+1:02d}.png", dpi=100, pad_inches=0
        )
        plt.close()

    # Plot combined figure
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        # Original - transpose for matplotlib (C, H, W) -> (H, W, C)
        orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Reconstructed - transpose for matplotlib
        recon_img = x_recon[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.suptitle("Validation Reconstructions")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/validation_reconstructions.png", bbox_inches="tight", dpi=600
    )
    plt.close()


if __name__ == "__main__":
    sns.set_context("talk", font_scale=1)
    sns.set_style("ticks")

    if not os.path.exists("training_output"):
        os.makedirs("training_output")
    output_dir = "training_output"

    # Create train, validation, and test datasets
    train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
        grid_index_csv="csv_data_files/grid_image_index.csv",
        cross_power_csv="csv_data_files/cross_power_law_results.csv",
        ink_ph_csv="csv_data_files/ink_ph_content_mmol_per_ml.csv",
        image_dir="colored_output",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        random_state=42,
    )

    # Train the model
    training_history = train_model(
        train_dataset,
        val_dataset,
        output_dir,
        epochs=200,
        lr=1e-3,
        batch_size=64,
        patience=15,
    )

    # Save the training history
    with open(f"{output_dir}/training_history.json", "w") as f:
        history_to_save = {
            "train_losses": [
                {
                    "epoch": i + 1,
                    "total_loss": float(L[0]),
                    "recon_loss": float(L[1]),
                    "kl_loss": float(L[2]),
                }
                for i, L in enumerate(training_history["train_losses"])
            ],
            "val_losses": [
                {
                    "epoch": i + 1,
                    "total_loss": float(L[0]),
                    "recon_loss": float(L[1]),
                    "kl_loss": float(L[2]),
                }
                for i, L in enumerate(training_history["val_losses"])
            ],
            "best_val_loss": float(training_history["best_val_loss"]),
            "best_epoch": int(training_history["best_epoch"]),
        }
        json.dump(history_to_save, f, indent=4)

    # Save scaler and test dataset information
    with open(f"{output_dir}/dataset_info.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": train_dataset.scaler,
                "test_dataset": test_dataset,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
            },
            f,
        )

    # Save scaler statistics
    if hasattr(train_dataset, "scaler") and train_dataset.scaler is not None:
        scaler_stats = {
            "mean": train_dataset.scaler.mean_.tolist(),
            "std": train_dataset.scaler.scale_.tolist(),
            "feature_names": [
                "intensity",
                "log10(eta0)",
                "log10(m)",
                "n",
                "log1p(ph_content)",
                "tan_delta",
            ],
        }
        with open(f"{output_dir}/scaler_statistics.json", "w") as f:
            json.dump(scaler_stats, f, indent=4)

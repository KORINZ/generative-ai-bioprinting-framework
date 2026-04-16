import torch
import numpy as np
import pandas as pd
import pickle
import re
import json
from datetime import datetime
from pathlib import Path
from PIL import Image

from model_training.beta_CVAE.train_beta_CVAE import BetaCVAE
from model_training.beta_CVAE.beta_CVAE_hyperparameters import LATENT_DIM, CONDITION_DIM


def load_model_and_scaler(model_path, scaler_path) -> tuple:
    """Load trained model and scaler"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BetaCVAE(latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, device


def get_ink_properties(ink_type_ph, rheology_df, ph_df) -> dict:
    """Get rheological properties and pH content for an ink type"""

    # Parse ink type
    ink_info = parse_ink_type(ink_type_ph)
    if ink_info is None:
        raise ValueError(f"Could not parse ink type: {ink_type_ph}")

    # Find rheology data
    rheology_match = find_rheology_match(ink_info, rheology_df)
    if rheology_match is None:
        raise ValueError(f"No rheology data found for ink type: {ink_type_ph}")

    # Get pH content
    ph_content = get_ph_content(ink_type_ph, ph_df)
    if ph_content is None:
        raise ValueError(f"No pH content found for ink type: {ink_type_ph}")

    return {
        "eta0": rheology_match["eta0"],
        "m": rheology_match["m"],
        "n": rheology_match["n"],
        "tan_delta": rheology_match["tan_delta"],
        "ph_content": ph_content,
    }


def parse_ink_type(ink_type_ph) -> dict | None:
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
        return {"type": "single", "ink_type": ink_type, "concentration": concentration}

    return None


def find_rheology_match(ink_info, rheology_df) -> pd.Series | None:
    """Find matching rheology data based on ink info"""

    if ink_info["type"] == "single":
        if ink_info["ink_type"] == "ALG":
            pattern = f"^ALG-I1G-Ph_{ink_info['concentration']}_"
            matches = rheology_df[
                (rheology_df["ink_type"].str.match(pattern))
                & (~rheology_df["ink_type"].str.contains("HA-HQ-Ph"))
            ]
        else:  # HA
            pattern = f"^HA-HQ-Ph_{ink_info['concentration']}_"
            matches = rheology_df[rheology_df["ink_type"].str.match(pattern)]
    else:  # composite
        pattern = f"ALG-I1G-Ph_{ink_info['alg_conc']}_HA-HQ-Ph_{ink_info['ha_conc']}_"
        matches = rheology_df[
            rheology_df["ink_type"].str.contains(pattern, regex=False)
        ]

    if len(matches) > 0:
        return matches.iloc[0]
    return None


def get_ph_content(ink_type_ph, ph_df) -> float | None:
    """Get Ph content for the ink type"""

    ph_match = ph_df[ph_df["ink_type_ph"] == ink_type_ph]
    if len(ph_match) > 0:
        return ph_match.iloc[0]["mmol_ph_ml"]
    return None


def generate_images(
    model,
    scaler,
    device,
    ink_properties,
    intensity,
    n_samples=5,
    temperature=1.0,
    seed=None,
) -> list:
    """Generate RGB images for given ink properties and intensity"""

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create condition vector
    condition = np.array(
        [
            intensity / 100.0,  # Normalize to [0, 1]
            np.log10(ink_properties["eta0"]),
            np.log10(ink_properties["m"]),
            ink_properties["n"],
            np.log1p(ink_properties["ph_content"]),
            ink_properties["tan_delta"],
        ]
    )

    # Apply scaling
    condition_scaled = scaler.transform(condition.reshape(1, -1))[0]
    condition_tensor = (
        torch.tensor(condition_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Repeat condition for batch generation
    conditions = condition_tensor.repeat(n_samples, 1)

    # Sample from latent space with temperature
    z = torch.randn(n_samples, LATENT_DIM).to(device) * temperature

    # Generate images
    with torch.no_grad():
        generated = model.decoder(z, conditions)

    # Convert to numpy - keep raw values from model
    images = generated.cpu().numpy()

    # Print value range info (only once)
    if n_samples > 0:
        all_values = images.flatten()
        print(
            f"    Raw model output range: [{all_values.min():.3f}, {all_values.max():.3f}]"
        )

    # Process images
    pil_images = []
    for i in range(n_samples):
        # Get 3-channel RGB output from model
        img_array = images[i]  # Shape: (3, H, W)

        # Transpose from (C, H, W) to (H, W, C)
        img_array = np.transpose(img_array, (1, 2, 0))

        # Clip to [0, 1] range and scale to 0-255
        img_array_clipped = np.clip(img_array, 0, 1)
        img_array_uint8 = (img_array_clipped * 255).astype(np.uint8)

        # Create RGB PIL image
        img = Image.fromarray(img_array_uint8, mode="RGB")

        # Resize to 800x800 using high-quality resampling
        img = img.resize((800, 800), Image.LANCZOS)
        pil_images.append(img)

    return pil_images


def main() -> None:
    """Main function to generate images for different ink types and intensities"""

    # Configuration
    ink_types = [
        "ALG1.0",
        "ALG3.0",
        "ALG5.0",
        "HA1.0",
        "HA2.0",
        "HA3.0",
        "ALG0.5HA0.5",
        "ALG1.0HA1.0",
        "ALG2.0HA2.0",
    ]
    intensities = [50]
    n_generations = 5
    temperature = 0.7

    # Paths
    model_path = "models/beta_CVAE/grid_model.pth"
    scaler_path = "models/beta_CVAE/condition_scaler.pkl"
    output_dir = Path("generated_images")

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"generation_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Load data
    model, scaler, device = load_model_and_scaler(model_path, scaler_path)
    rheology_df = pd.read_csv("csv_data_files_generation/cross_power_law_results.csv")
    ph_df = pd.read_csv("csv_data_files_generation/ink_ph_content_mmol_per_ml.csv")

    print(f"\nGenerating RGB images with:")
    print(f"- Temperature: {temperature}")
    print(f"- Images per condition: {n_generations}")
    print(f"- Output directory: {run_dir}")

    # Generate images for each ink type and intensity
    for ink_type in ink_types:
        print(f"\nProcessing {ink_type}...")

        try:
            # Get ink properties
            ink_props = get_ink_properties(ink_type, rheology_df, ph_df)

            # Create directory for this ink type
            ink_dir = run_dir / ink_type
            ink_dir.mkdir(exist_ok=True)

            for intensity in intensities:
                print(f"  Generating at intensity {intensity}%...")

                # Generate images
                images = generate_images(
                    model,
                    scaler,
                    device,
                    ink_props,
                    intensity,
                    n_samples=n_generations,
                    temperature=temperature,
                )

                # Save images
                for i, img in enumerate(images):
                    filename = (
                        ink_dir / f"{ink_type}_intensity{intensity}_sample{i+1}.png"
                    )
                    img.save(filename)

            print(
                f"  ✓ Generated {len(intensities) * n_generations} images for {ink_type}"
            )

        except Exception as e:
            print(f"  ✗ Error processing {ink_type}: {str(e)}")
            continue

    # Save generation parameters
    params = {
        "timestamp": timestamp,
        "ink_types": ink_types,
        "intensities": intensities,
        "n_generations": n_generations,
        "temperature": temperature,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    with open(run_dir / "generation_params.json", "w") as f:
        json.dump(params, f, indent=4)

    print(f"\n✓ Generation complete! Images saved to {run_dir}")
    print(
        f"Total images generated: {len(ink_types) * len(intensities) * n_generations}"
    )


if __name__ == "__main__":
    main()

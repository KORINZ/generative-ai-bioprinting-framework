import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import pickle
import os
import re

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score


# Set up plotting style
sns.set_context("talk", font_scale=1.45)
sns.set_style("ticks")


CROSS_VALIDATION_FOLDS = 10


# Read and preprocess data
data_path = "cross_power_law_results.csv"
data = pd.read_csv(data_path)


def extract_concentrations(ink_type_str) -> pd.Series:
    """Extract ALG and HA concentrations from the new ink type format."""

    alg = 0.0
    ha = 0.0

    # Check for ALG concentration
    alg_match = re.search(r"ALG-I1G-Ph_(\d+\.?\d*)", ink_type_str)
    if alg_match:
        alg = float(alg_match.group(1))

    # Check for HA concentration
    ha_match = re.search(r"HA-HQ-Ph_(\d+\.?\d*)", ink_type_str)
    if ha_match:
        ha = float(ha_match.group(1))

    # Special case: if starts with HA-HQ-Ph and no ALG found
    if ink_type_str.startswith("HA-HQ-Ph_") and alg == 0.0:
        ha_only_match = re.search(r"HA-HQ-Ph_(\d+\.?\d*)", ink_type_str)
        if ha_only_match:
            ha = float(ha_only_match.group(1))

    return pd.Series({"ALG": alg, "HA": ha})


# Extract ALG and HA concentrations
data[["ALG", "HA"]] = data["ink_type"].apply(extract_concentrations)

# Drop rows with missing values
data = data.dropna()

# Prepare input features and target variables (including tan_delta)
X = data[["ALG", "HA"]].values
Y = data[["eta0", "m", "n", "tan_delta"]].values

# Apply log transformation to Y (except for n)
Y_transformed = Y.copy()
Y_transformed[:, [0, 1, 3]] = np.log(
    Y[:, [0, 1, 3]]
)  # Log transform eta0, m, tan_delta
# n (index 2) remains untransformed
Y = Y_transformed

# Scale the features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Initialize scalers for each target variable
scalers_Y = [StandardScaler() for _ in range(4)]


def train_svr_cv(X: np.ndarray, Y: np.ndarray) -> tuple[SVR, np.float64, dict]:
    """Train an SVR model using grid search with cross-validation."""

    param_grid = {
        "C": np.linspace(10, 5000, 7),
        "gamma": np.linspace(0.01, 0.1, 7),
        "epsilon": np.linspace(0.01, 0.1, 7),
    }

    svr_grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=CROSS_VALIDATION_FOLDS,
        verbose=2,
        n_jobs=-1,
    )

    svr_grid.fit(X, Y)

    best_model = svr_grid.best_estimator_
    best_params = svr_grid.best_params_

    mse_scores = -cross_val_score(
        best_model, X, Y, scoring="neg_mean_squared_error", cv=CROSS_VALIDATION_FOLDS
    )
    mean_mse = np.mean(mse_scores).round(4)

    return best_model, mean_mse, best_params


# Train models for all target variables
svr_models = []
mse_scores = []
best_params_list = []
rmse_scores_original_space = []

for i, name in enumerate(["eta0", "m", "n", "tan_delta"]):
    y = Y[:, i].reshape(-1, 1)
    y_scaled = scalers_Y[i].fit_transform(y)

    svr_model, mse_score, best_params = train_svr_cv(X_scaled, y_scaled.ravel())
    svr_models.append(svr_model)
    mse_scores.append(mse_score)
    best_params_list.append(best_params)

    # Calculate RMSE in original space
    y_pred_scaled = cross_val_predict(
        svr_model, X_scaled, y_scaled.ravel(), cv=CROSS_VALIDATION_FOLDS
    )
    y_pred = scalers_Y[i].inverse_transform(y_pred_scaled.reshape(-1, 1))

    # Only apply exp transformation for log-transformed variables
    if name != "n":
        y_pred = np.exp(y_pred)
        y_actual = np.exp(y)
    else:
        y_actual = y

    rmse_original = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    rmse_scores_original_space.append(rmse_original)

print(f"Total data points: {len(data)}")
for i, name in enumerate(["eta0", "m", "n", "tan_delta"]):
    print(f"Best parameters for SVR model for {name}: {best_params_list[i]}")
    print(
        f"Cross-validated Root Mean Squared Error for {name} in original space: {rmse_scores_original_space[i]:.3f}"
    )


def log_tick_formatter(val) -> str:
    """Custom formatter for log scale ticks to display as powers of 10."""

    return f"$10^{{{val:g}}}$"


def plot_3d_surface(
    model, title, scaler_X, scaler_Y, data, index, save_fig=False
) -> None:
    """Plot a 3D surface plot of the predicted values for a given feature."""

    alg_range = np.linspace(data["ALG"].min(), data["ALG"].max(), 25)
    ha_range = np.linspace(data["HA"].min(), data["HA"].max(), 25)
    alg_grid, ha_grid = np.meshgrid(alg_range, ha_range)

    grid_scaled = scaler_X.transform(
        np.column_stack((alg_grid.ravel(), ha_grid.ravel()))
    )
    predictions_scaled = model.predict(grid_scaled)
    predictions_transformed = scaler_Y.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    )[:, 0]

    # Only apply exp transformation for log-transformed variables
    if title != "n":
        predictions_original = np.exp(predictions_transformed).reshape(alg_grid.shape)
        original_data_points = np.exp(Y[:, index])
    else:
        predictions_original = predictions_transformed.reshape(alg_grid.shape)
        original_data_points = Y[:, index]

    # Apply log10 transformation for eta0, m, and tan_delta
    use_log_scale = title in ["eta0", "m", "tan_delta"]
    if use_log_scale:
        predictions_plot = np.log10(predictions_original)
        data_points_z = np.log10(original_data_points)
    else:
        predictions_plot = predictions_original
        data_points_z = original_data_points

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d", computed_zorder=False)

    if title == "eta0":
        cmap = "viridis"
        zlabel = r"$\eta_{0}$ [Pa·s]"
        marker = "o"
        ax.set_title(f"Zero-Shear-Rate Viscosity" + r" ($\eta_{0}$)")
    elif title == "m":
        cmap = "plasma"
        zlabel = r"$m$ [s]"
        marker = "s"
        ax.set_title(f"Time Constant" + r" ($m$)")
    elif title == "n":
        cmap = "coolwarm"
        zlabel = r"$n$ [-]"
        marker = "^"
        ax.set_title(f"Shear-Thinning Index" + r" ($n$)")
    elif title == "tan_delta":
        cmap = "cividis"
        zlabel = r"$\tan(\delta)$ [-]"
        marker = "D"
        ax.set_title(f"Loss Tangent" + r" ($\tan(\delta)$)")
        # reverse z-axis for tan_delta
        ax.invert_zaxis()

    surf = ax.plot_surface(
        alg_grid, ha_grid, predictions_plot, cmap=cmap, alpha=0.8, linewidth=0.25
    )
    ax.scatter(
        data["ALG"],
        data["HA"],
        data_points_z,
        c="red",
        marker=marker,
        s=100,
        edgecolor="black",
        label="Experimental Data" if title == "tan_delta" else "Fitted Empirical Data",
        depthshade=False,
    )

    ax.set_xlabel("ALG-Ph [% (w/v)]", fontweight="bold")
    ax.set_ylabel("HA-Ph [% (w/v)]", fontweight="bold")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, fontweight="bold", rotation=90)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        fancybox=False,
        framealpha=1,
        edgecolor="black",
    )

    # Set y-axis (HA) ticks
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticklabels(
        [f"{i:.1f}" for i in np.arange(0, 3.1, 0.5)],
    )

    # Set x-axis (ALG) ticks
    ax.set_xticks([0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ax.set_xticklabels(
        [f"{i:.1f}" for i in np.arange(0, 5.1, 1.0)],
    )

    # Set z-ticks and formatter for log scale plots
    if use_log_scale:
        if title == "eta0":
            # Determine appropriate range based on data
            z_min = np.floor(np.log10(original_data_points.min()))
            z_max = np.ceil(np.log10(original_data_points.max()))
            z_ticks = np.arange(z_min, z_max + 1)
            ax.set_zticks(z_ticks)
        elif title == "m":
            # Determine appropriate range based on data
            z_min = np.floor(np.log10(original_data_points.min()))
            z_max = np.ceil(np.log10(original_data_points.max()))
            z_ticks = np.arange(z_min, z_max + 1)
            ax.set_zticks(z_ticks)
        elif title == "tan_delta":
            # Determine appropriate range based on data
            z_min = np.floor(np.log10(original_data_points.min()))
            z_max = np.ceil(np.log10(original_data_points.max()))
            z_ticks = np.arange(z_min, z_max + 1)
            ax.set_zticks(z_ticks)

        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    else:
        # For n, use default ticks
        ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # Add padding to labels
    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 18
    ax.zaxis.labelpad = 22

    # Add padding to tick labels
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.tick_params(axis="z", pad=10)

    ax.view_init(25, -135)

    if save_fig:
        if not os.path.exists(f"SVR/images"):
            os.makedirs(f"SVR/images")
        plt.savefig(
            f"SVR/images/{title}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.48,
        )

    plt.show()


def plot_actual_vs_predicted_cv(
    model, feature_index, scaler_Y, X, Y, title, save_fig=False
) -> None:
    """Plot actual vs. predicted values using cross-validation predictions."""

    # Get the target variable
    y = Y[:, feature_index].reshape(-1, 1)

    # Scale the target variable
    y_scaled = scaler_Y.fit_transform(y)

    # Predict using cross-validation on scaled data
    Y_pred_scaled = cross_val_predict(
        model, X, y_scaled.ravel(), cv=CROSS_VALIDATION_FOLDS
    )

    # Inverse transform the predictions
    Y_pred_transformed = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1))

    # Convert from log space to original space (except for n)
    if title != "n":
        Y_pred = np.exp(Y_pred_transformed)
        Y_actual = np.exp(y)
    else:
        Y_pred = Y_pred_transformed
        Y_actual = y

    # Calculate metrics
    r2 = r2_score(Y_actual, Y_pred)
    mae = np.mean(np.abs(Y_actual - Y_pred))
    rmse = np.sqrt(np.mean((Y_actual - Y_pred) ** 2))

    plt.figure(figsize=(7.8, 7.8))

    if title == "eta0":
        unit = "Pa·s"
        marker = "o"
        plt.title("Zero-Shear-Rate Viscosity" + r" ($\eta_{0}$)")
    elif title == "m":
        unit = "s"
        marker = "s"
        plt.title("Time Constant" + r" ($m$)")
    elif title == "n":
        unit = "-"
        marker = "^"
        plt.title("Shear-Thinning Index" + r" ($n$)")
    elif title == "tan_delta":
        unit = "-"
        marker = "D"
        plt.title("Loss Tangent" + r" ($\tan(\delta)$)")

    # Plot
    plt.scatter(Y_actual, Y_pred, alpha=0.5, s=100, marker=marker, color="red")
    plt.plot(
        [Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], "k--", lw=2
    )
    plt.xlabel(f"Actual [{unit}]", fontweight="bold")
    plt.ylabel(f"Predicted [{unit}]", fontweight="bold")

    if title == "n" or title == "tan_delta":
        unit = ""

    # Create a box with error evaluation results
    textstr = "\n".join(
        [f"R-squared: {r2:.3f}", f"MAE: {mae:.3f} {unit}", f"RMSE: {rmse:.3f} {unit}"]
    )
    props = dict(facecolor="white", alpha=1, edgecolor="black")
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=props,
    )

    # log scale
    if title == "eta0" or title == "m" or title == "tan_delta":
        plt.xscale("log")
        plt.yscale("log")

    # plt.locator_params(axis="both", nbins=6)

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            f"SVR/images/{title}_actual_vs_predicted.png",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


# Plot 3D surfaces
for i, name in enumerate(["eta0", "m", "n", "tan_delta"]):
    plot_3d_surface(svr_models[i], name, scaler_X, scalers_Y[i], data, i, save_fig=True)
    plot_actual_vs_predicted_cv(
        svr_models[i], i, scalers_Y[i], X_scaled, Y, name, save_fig=True
    )

# Save models and scalers
if not os.path.exists(f"SVR/model"):
    os.makedirs(f"SVR/model")

for i, name in enumerate(["eta0", "m", "n", "tan_delta"]):
    with open(f"SVR/model/svr_{name}_model.pkl", "wb") as file:
        pickle.dump(svr_models[i], file)

with open(f"SVR/model/scaler_X.pkl", "wb") as file:
    pickle.dump(scaler_X, file)

for i, name in enumerate(["eta0", "m", "n", "tan_delta"]):
    with open(f"SVR/model/scaler_Y_{name}.pkl", "wb") as file:
        pickle.dump(scalers_Y[i], file)

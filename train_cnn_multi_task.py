import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")

np.random.seed(42)
tf.random.set_seed(42)

REGRESSION_TARGETS = [
    "construct_area_mm2",
    "average_filament_width_mm",
    "average_hd",
    "dice_coefficient",
]
CLASSIFICATION_TARGET = "is_complete"


class MultiOutputModel:
    """Multi-task CNN for regression and classification of bioprinted construct images."""

    def __init__(self, image_size=(128, 128)) -> None:
        self.image_size = image_size
        self.model = None
        self.scalers = {target: MinMaxScaler() for target in REGRESSION_TARGETS}

    def rotate_image(self, image, angle) -> np.ndarray:
        """Rotate image by specified angle (0, 90, 180, 270 degrees)."""

        if angle == 0:
            return image
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def load_and_preprocess_image(self, image_path, rotation_angle=0) -> np.ndarray:
        """Load image, convert to grayscale, resize, rotate if needed, and normalize."""

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.resize(img, self.image_size)
        if rotation_angle != 0:
            img = self.rotate_image(img, rotation_angle)
        img = img / 255.0
        return img.reshape((*self.image_size, 1))

    def prepare_data(self, images_dir, csv_data) -> tuple:
        """Load images and corresponding regression/classification targets, split into train/val/test, and apply scaling and augmentation."""

        X = []
        y_regression = {target: [] for target in REGRESSION_TARGETS}
        y_classification = []

        images_path = Path(images_dir)

        for _, row in csv_data.iterrows():
            image_name = row["image_name"]
            ink_type = row["ink_type"]
            image_path = images_path / ink_type / image_name

            if image_path.exists():
                try:
                    img = self.load_and_preprocess_image(image_path)
                    X.append(img)
                    for target in REGRESSION_TARGETS:
                        y_regression[target].append(row[target])
                    y_classification.append(1 if row[CLASSIFICATION_TARGET] else 0)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            else:
                print(f"Image not found: {image_path}")

        if not X:
            raise ValueError("No valid images found in the dataset")

        X = np.array(X)
        y_regression = {k: np.array(v) for k, v in y_regression.items()}
        y_classification = np.array(y_classification)

        # Train/val/test split (70/20/10)
        indices = np.arange(len(X))
        train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 / 3, random_state=42)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train_reg = {k: v[train_idx] for k, v in y_regression.items()}
        y_val_reg = {k: v[val_idx] for k, v in y_regression.items()}
        y_test_reg = {k: v[test_idx] for k, v in y_regression.items()}
        y_train_cls = y_classification[train_idx]
        y_val_cls = y_classification[val_idx]
        y_test_cls = y_classification[test_idx]

        # Fit scalers on training data and transform all splits
        y_train_scaled, y_val_scaled, y_test_scaled = {}, {}, {}
        for target in REGRESSION_TARGETS:
            y_train_scaled[target] = (
                self.scalers[target]
                .fit_transform(y_train_reg[target].reshape(-1, 1))
                .ravel()
            )
            y_val_scaled[target] = (
                self.scalers[target].transform(y_val_reg[target].reshape(-1, 1)).ravel()
            )
            y_test_scaled[target] = (
                self.scalers[target]
                .transform(y_test_reg[target].reshape(-1, 1))
                .ravel()
            )

        # Augment training data only with rotations
        X_train_aug = []
        y_train_aug = {target: [] for target in REGRESSION_TARGETS}
        y_train_cls_aug = []
        rotation_angles = [0, 90, 180, 270]

        for i in range(len(X_train)):
            for angle in rotation_angles:
                if angle == 0:
                    img = X_train[i]
                else:
                    img = self.rotate_image(X_train[i], angle)
                    img = img.reshape((*self.image_size, 1))
                # Original
                X_train_aug.append(img)
                for target in REGRESSION_TARGETS:
                    y_train_aug[target].append(y_train_scaled[target][i])
                y_train_cls_aug.append(y_train_cls[i])

        X_train_aug = np.array(X_train_aug)
        y_train_aug = {k: np.array(v) for k, v in y_train_aug.items()}
        y_train_cls_aug = np.array(y_train_cls_aug)

        # Stack regression targets
        y_train_reg_stacked = np.column_stack(
            [y_train_aug[t] for t in REGRESSION_TARGETS]
        )
        y_val_reg_stacked = np.column_stack(
            [y_val_scaled[t] for t in REGRESSION_TARGETS]
        )
        y_test_reg_stacked = np.column_stack(
            [y_test_scaled[t] for t in REGRESSION_TARGETS]
        )

        return (
            X_train_aug,
            X_val,
            X_test,
            y_train_reg_stacked,
            y_val_reg_stacked,
            y_test_reg_stacked,
            y_train_cls_aug,
            y_val_cls,
            y_test_cls,
        )

    def build_model(self) -> None:
        """Build the multi-task CNN architecture with shared convolutional layers and separate regression/classification heads."""

        regularizer = tf.keras.regularizers.l2(1e-4)

        inputs = Input(shape=(*self.image_size, 1))

        # Shared convolutional backbone (reduced complexity)
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizer,
            padding="same",
        )(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizer,
            padding="same",
        )(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizer,
            padding="same",
        )(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)

        # Shared dense layers
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizer)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)

        # Regression head (4 outputs)
        reg_branch = layers.Dense(
            64, activation="relu", kernel_regularizer=regularizer
        )(x)
        regression_output = layers.Dense(len(REGRESSION_TARGETS))(reg_branch)

        # Classification head (binary)
        cls_branch = layers.Dense(
            32, activation="relu", kernel_regularizer=regularizer
        )(x)
        classification_output = layers.Dense(1, activation="sigmoid")(cls_branch)

        self.model = Model(
            inputs=inputs, outputs=[regression_output, classification_output]
        )

        self.model.compile(
            optimizer="adam",
            loss=["mse", "binary_crossentropy"],
            loss_weights=[1.0, 0.5],
            metrics=[["mae"], ["accuracy"]],
        )

        self.model.summary()

    def train(
        self,
        X_train,
        y_train_reg,
        y_train_cls,
        X_val,
        y_val_reg,
        y_val_cls,
        epochs=500,
        batch_size=32,
    ) -> tf.keras.callbacks.History:
        """Train the multi-task CNN with early stopping and learning rate reduction on plateau."""

        # Reshape classification targets to (N, 1) to match sigmoid output
        y_train_cls = y_train_cls.reshape(-1, 1).astype(np.float32)
        y_val_cls = y_val_cls.reshape(-1, 1).astype(np.float32)

        return self.model.fit(
            X_train,
            [y_train_reg, y_train_cls],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, [y_val_reg, y_val_cls]),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=20, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
                ),
            ],
        )

    def plot_training_history(self, history, output_dir="CNNs") -> None:
        """Plot training and validation loss curves for regression and classification tasks."""

        Path(output_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        best_epoch = np.argmin(history.history["val_loss"])

        # Find the regression and classification loss keys
        hist_keys = list(history.history.keys())
        reg_loss_key = [
            k for k in hist_keys if "loss" in k and "val" not in k and k != "loss"
        ][0]
        cls_loss_key = [
            k for k in hist_keys if "loss" in k and "val" not in k and k != "loss"
        ][1]
        val_reg_loss_key = "val_" + reg_loss_key
        val_cls_loss_key = "val_" + cls_loss_key

        # Find accuracy key
        acc_key = [k for k in hist_keys if "accuracy" in k and "val" not in k][0]
        val_acc_key = "val_" + acc_key

        # Regression loss
        ax = axes[0]
        ax.plot(history.history[reg_loss_key], label="Training", color="tab:blue")
        ax.plot(
            history.history[val_reg_loss_key], label="Validation", color="tab:orange"
        )
        ax.axvline(
            x=best_epoch,
            color="gray",
            linestyle="--",
            label=f"Best (Epoch {best_epoch})",
        )
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss (MSE)", fontweight="bold")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Regression Loss")

        # Classification loss
        ax = axes[1]
        ax.plot(history.history[cls_loss_key], label="Training", color="tab:blue")
        ax.plot(
            history.history[val_cls_loss_key], label="Validation", color="tab:orange"
        )
        ax.axvline(x=best_epoch, color="gray", linestyle="--")
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss (BCE)", fontweight="bold")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Classification Loss")

        # Classification accuracy
        ax = axes[2]
        ax.plot(history.history[acc_key], label="Training", color="tab:blue")
        ax.plot(history.history[val_acc_key], label="Validation", color="tab:orange")
        ax.axvline(x=best_epoch, color="gray", linestyle="--")
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Accuracy", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Classification Accuracy")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/multioutput_training_history.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def evaluate_predictions(self, X, y_reg, y_cls, output_dir="CNNs") -> tuple:
        """Evaluate model predictions on test set, compute regression metrics and classification accuracy, and plot results."""

        Path(output_dir).mkdir(exist_ok=True)

        predictions = self.model.predict(X)
        pred_reg, pred_cls = predictions

        # Inverse transform regression predictions
        y_actual_reg = {}
        y_pred_reg = {}
        for i, target in enumerate(REGRESSION_TARGETS):
            y_actual_reg[target] = (
                self.scalers[target]
                .inverse_transform(y_reg[:, i].reshape(-1, 1))
                .ravel()
            )
            y_pred_reg[target] = (
                self.scalers[target]
                .inverse_transform(pred_reg[:, i].reshape(-1, 1))
                .ravel()
            )

        # Classification
        pred_cls_binary = (pred_cls.ravel() > 0.5).astype(int)
        cls_accuracy = accuracy_score(y_cls, pred_cls_binary)

        # Plot regression results
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()

        plot_titles = {
            "construct_area_mm2": r"Construct Area [mm$^2$]",
            "average_filament_width_mm": "Average Filament Width [mm]",
            "average_hd": "Average HD [-]",
            "dice_coefficient": "Dice Coefficient [-]",
        }
        markers = ["o", "s", "^", "D"]

        metrics = {}
        for i, target in enumerate(REGRESSION_TARGETS):
            ax = axes[i]
            y_act = y_actual_reg[target]
            y_prd = y_pred_reg[target]

            r2 = r2_score(y_act, y_prd)
            mae = np.mean(np.abs(y_act - y_prd))
            rmse = np.sqrt(np.mean((y_act - y_prd) ** 2))
            metrics[target] = {"r2": r2, "mae": mae, "rmse": rmse}

            ax.scatter(
                y_act,
                y_prd,
                alpha=0.6,
                s=50,
                color="steelblue",
                marker=markers[i],
                label="Data",
            )
            ax.plot(
                [y_act.min(), y_act.max()],
                [y_act.min(), y_act.max()],
                "r--",
                lw=2,
                label="Ideal",
            )

            ax.set_xlabel("Actual", fontweight="bold")
            ax.set_ylabel("Predicted", fontweight="bold")

            textstr = f"R²: {r2:.3f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
            props = dict(facecolor="white", alpha=0.9, edgecolor="black")
            ax.text(
                0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=props,
                fontsize=10,
            )

            ax.grid(True, alpha=0.3)
            ax.set_title(plot_titles[target])
            ax.legend(fontsize=12, loc="lower right")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/multioutput_regression_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Classification confusion matrix
        print(f"\nClassification Accuracy (is_complete): {cls_accuracy:.3f}")
        print(f"  True positives: {np.sum((y_cls == 1) & (pred_cls_binary == 1))}")
        print(f"  True negatives: {np.sum((y_cls == 0) & (pred_cls_binary == 0))}")
        print(f"  False positives: {np.sum((y_cls == 0) & (pred_cls_binary == 1))}")
        print(f"  False negatives: {np.sum((y_cls == 1) & (pred_cls_binary == 0))}")

        # Plot confusion matrix
        cm = confusion_matrix(y_cls, pred_cls_binary)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Incomplete", "Complete"],
            yticklabels=["Incomplete", "Complete"],
            annot_kws={"size": 14},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual", fontweight="bold")
        ax.set_title("Classification Confusion Matrix (is_complete)")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/multioutput_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return metrics, cls_accuracy

    def predict_single(self, image_path) -> dict:
        """Predict regression and classification outputs for a single image."""

        img = self.load_and_preprocess_image(image_path)
        predictions = self.model.predict(np.array([img]), verbose=0)
        pred_reg, pred_cls = predictions

        results = {}
        for i, target in enumerate(REGRESSION_TARGETS):
            results[target] = self.scalers[target].inverse_transform(
                [[pred_reg[0, i]]]
            )[0, 0]
        results[CLASSIFICATION_TARGET] = pred_cls[0, 0] > 0.5

        return results


def main() -> None:
    """Main function to load data, train multi-task CNN, and evaluate results."""

    # Load analysis results CSV
    csv_path = "csv_data_files/generated_images_analysis_auto_2.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Please run batch_analysis.py first to generate the analysis results.")
        return

    csv_data = pd.read_csv(csv_path)
    print(f"Loaded {len(csv_data)} samples from CSV")

    # Initialize model
    model = MultiOutputModel(image_size=(128, 128))

    # Prepare data (70/20/10 split)
    images_dir = "generated_images/generation_20260211_040537"
    (
        X_train,
        X_val,
        X_test,
        y_train_reg,
        y_val_reg,
        y_test_reg,
        y_train_cls,
        y_val_cls,
        y_test_cls,
    ) = model.prepare_data(images_dir, csv_data)

    print(f"\nTraining samples (after augmentation): {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Build and train model
    model.build_model()
    history = model.train(
        X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls
    )

    # Plot training history
    model.plot_training_history(history)

    # Save training history for reproducibility
    joblib.dump(history.history, "CNNs/training_history.pkl")
    print("Training history saved to CNNs/training_history.pkl")

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    metrics, cls_acc = model.evaluate_predictions(X_test, y_test_reg, y_test_cls)

    for target, m in metrics.items():
        print(f"{target}: R²={m['r2']:.3f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")

    # Save model and scalers
    save_dir = Path("CNNs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.model.save(str(save_dir / "multioutput_model.keras"))
    joblib.dump(model.scalers, str(save_dir / "multioutput_scalers.pkl"))
    print(f"\nModel and scalers saved to {save_dir}/")


if __name__ == "__main__":
    main()

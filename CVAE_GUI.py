"""
β-CVAE Bioprinting Image Generator GUI

This application allows users to generate bioprinting images using a trained
β-CVAE model. Users can either specify ALG-Ph and HA-Ph concentrations (with
SVR-predicted rheological parameters) or manually adjust parameters.

Additionally displays CNN-predicted printability metrics for grid models.
"""

import customtkinter as ctk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tkinter import filedialog, messagebox
import pickle
import joblib
import json
from pathlib import Path
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

# Constants for Ph content calculation
MMOL_PER_GRAM_ALG = 0.124
MMOL_PER_GRAM_HA = 0.294


def render_latex_label(latex_str, fontsize=14, text_color="black"):
    """Render LaTeX string to CTkImage for use in labels."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1), dpi=100)
    ax.axis("off")
    text = ax.text(
        0.5,
        0.5,
        latex_str,
        fontsize=fontsize,
        color=text_color,
        verticalalignment="center",
        horizontalalignment="center",
        multialignment="center",
    )
    fig.canvas.draw()
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox = bbox.expanded(1.1, 1.2)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches=bbox_inches, transparent=True, dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    plt.close(fig)
    return ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))


# CVAE Model architecture
class Encoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        self.condition_dim = condition_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc_condition = nn.Linear(condition_dim, 128)
        self.fc_mu = nn.Linear(16384 + 128, latent_dim)
        self.fc_logvar = nn.Linear(16384 + 128, latent_dim)

    def forward(self, x, c):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = h.view(h.size(0), -1)
        c_encoded = F.relu(self.fc_condition(c))
        h = torch.cat([h, c_encoded], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        self.condition_dim = condition_dim
        self.fc_condition = nn.Linear(condition_dim, 128)
        self.fc = nn.Linear(latent_dim + 128, 256 * 8 * 8)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, z, c):
        c_encoded = F.relu(self.fc_condition(c))
        h = torch.cat([z, c_encoded], dim=1)
        h = F.relu(self.fc(h))
        h = h.view(h.size(0), 256, 8, 8)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        x_recon = torch.sigmoid(self.conv4(h))
        return x_recon


class BetaCVAE(nn.Module):
    def __init__(self, latent_dim=32, condition_dim=6, beta=5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, condition_dim)

    def generate(self, c, temperature=1.0):
        z = torch.randn(c.size(0), self.latent_dim, device=c.device) * temperature
        return self.decoder(z, c)


class CVAEGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("β-CVAE Bioprinting Image Generator")
        self.geometry("1280x900")
        self.minsize(1000, 600)
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("customtkinter_theme/orange-light.json")

        # set iconbitmap
        icon_path = "icons/icon.png"
        # Convert png to ico if necessary
        if Path(icon_path).exists():
            try:
                import os
                from PIL import Image

                if os.name == "nt":
                    ico_path = "icons/icon.ico"
                    img = Image.open(icon_path)
                    img.save(ico_path, format="ICO", sizes=[(32, 32)])
                    self.iconbitmap(ico_path)

                # Convert to gif for posix systems
                elif os.name == "posix":
                    gif_path = "icons/icon.gif"
                    img = Image.open(icon_path)
                    img.save(gif_path, format="GIF")

                    import tkinter as tk

                    img_tk = tk.PhotoImage(file=gif_path)
                    self.call("wm", "iconphoto", self._w, img_tk)

            except Exception as e:
                print(f"Warning: Could not set icon: {e}")

        # Parameter ranges
        self.param_ranges = {
            "intensity": (0, 100),
            "eta_0": (2.5e-2, 1.5e3),
            "m": (3.0e-4, 6.0),
            "n": (0.60, 0.70),
            "tan_delta": (0.5, 35.0),
            "D_ph": (0, 1.5e-2),
        }

        # Concentration ranges
        self.alg_range = (0, 5.0)
        self.ha_range = (0, 3.0)

        # Default values of ALG1.0 at 50% intensity
        self.defaults = {
            "intensity": 50,
            "eta_0": 0.08405,
            "m": 0.0006739,
            "n": 0.6369,
            "tan_delta": 23.0736,
            "D_ph": 0.001240,
        }

        self.model = None
        self.cvae_scaler = None
        self.svr_models = None
        self.cnn_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generated_image = None
        self.generated_image_array = None
        self.concentration_mode = True
        self.current_model_path = "models/beta_CVAE/grid_model.pth"
        self.cnn_enabled = True
        self.updating_from_concentration = False
        self.updating_from_rheology = False

        self.setup_ui()
        self.load_models()

        # Bind resize event
        self.bind("<Configure>", self.on_resize)
        self._last_size = (0, 0)

        # Bind click to unfocus entry widgets
        self.bind_all("<Button-1>", self.on_click_unfocus)

    def on_click_unfocus(self, event):
        """Unfocus entry widgets when clicking elsewhere."""
        widget = event.widget
        # Check for CTkEntry or internal tkinter Entry widget
        widget_class = widget.winfo_class()
        if widget_class not in ("Entry", "TEntry"):
            self.focus_set()

    def setup_ui(self):
        # Main container with grid layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=0, minsize=480)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Left panel for controls (fixed width)
        self.left_panel = ctk.CTkScrollableFrame(self.main_frame, width=500)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10), pady=0)

        # Right panel for image display (expandable)
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=0)

        self.setup_left_panel()
        self.setup_right_panel()

    def setup_left_panel(self):

        # Model loading section
        model_frame = ctk.CTkFrame(self.left_panel)
        model_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            model_frame,
            text="Current β-CVAE Model",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(5, 5))

        self.model_name_label = ctk.CTkLabel(
            model_frame,
            text=Path(self.current_model_path).name,
            text_color="gray",
            wraplength=400,
            font=ctk.CTkFont(size=16),
        )
        self.model_name_label.pack(pady=2)

        self.load_model_btn = ctk.CTkButton(
            model_frame,
            text="Load Model",
            command=self.load_different_model,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.load_model_btn.pack(pady=5)

        # Separator
        sep0 = ctk.CTkFrame(self.left_panel, height=2, fg_color="gray50")
        sep0.pack(fill="x", padx=10, pady=10)

        # Concentration input section
        conc_frame = ctk.CTkFrame(self.left_panel)
        conc_frame.pack(fill="x", padx=10, pady=5)

        conc_label = ctk.CTkLabel(
            conc_frame,
            text="Ink Compositions",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        conc_label.pack(pady=(5, 10))

        # ALG-Ph concentration with slider
        alg_frame = ctk.CTkFrame(conc_frame)
        alg_frame.pack(fill="x", padx=5, pady=2)
        self.alg_latex_img = render_latex_label(
            r"$\mathrm{ALG\text{-}Ph}\text{ [\% (w/v)]}$", fontsize=11
        )
        ctk.CTkLabel(
            alg_frame, text="", image=self.alg_latex_img, width=150, anchor="center"
        ).pack(side="left", padx=5)
        self.alg_entry = ctk.CTkEntry(alg_frame, width=70, font=ctk.CTkFont(size=14))
        self.alg_entry.pack(side="left", padx=5)
        self.alg_entry.insert(0, "1.0")
        self.alg_slider = ctk.CTkSlider(
            alg_frame, from_=0, to=5.0, width=150, number_of_steps=100
        )
        self.alg_slider.set(1.0)
        self.alg_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.alg_slider.configure(command=self.on_alg_slider_change)
        self.alg_entry.bind("<Return>", self.on_alg_entry_confirm)

        # HA-Ph concentration with slider
        ha_frame = ctk.CTkFrame(conc_frame)
        ha_frame.pack(fill="x", padx=5, pady=2)
        self.ha_latex_img = render_latex_label(
            r"$\mathrm{HA\text{-}Ph}\text{ [\% (w/v)]}$", fontsize=11
        )
        ctk.CTkLabel(
            ha_frame, text="", image=self.ha_latex_img, width=150, anchor="center"
        ).pack(side="left", padx=5)
        self.ha_entry = ctk.CTkEntry(ha_frame, width=70, font=ctk.CTkFont(size=14))
        self.ha_entry.pack(side="left", padx=5)
        self.ha_entry.insert(0, "0")
        self.ha_slider = ctk.CTkSlider(
            ha_frame, from_=0, to=3.0, width=150, number_of_steps=60
        )
        self.ha_slider.set(0)
        self.ha_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.ha_slider.configure(command=self.on_ha_slider_change)
        self.ha_entry.bind("<Return>", self.on_ha_entry_confirm)

        # Mode indicator
        self.mode_label = ctk.CTkLabel(
            conc_frame,
            text="Mode: SVR Rheological Predictors",
            text_color="green",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.mode_label.pack(pady=5)

        # Separator
        sep = ctk.CTkFrame(self.left_panel, height=2, fg_color="gray50")
        sep.pack(fill="x", padx=10, pady=10)

        # Parameter sliders section
        param_frame = ctk.CTkFrame(self.left_panel)
        param_frame.pack(fill="x", padx=10, pady=5)

        param_label = ctk.CTkLabel(
            param_frame,
            text="Printing and Rheological Parameters",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        param_label.pack(pady=(5, 10))

        self.param_widgets = {}
        self.create_parameter_control(
            param_frame,
            "intensity",
            "E_e [%]",
            0,
            100,
            self.defaults["intensity"],
            latex_label=r"$E_e$ [%]",
        )
        self.create_parameter_control(
            param_frame,
            "eta_0",
            "η₀ [Pa·s]",
            2.5e-2,
            1.5e3,
            self.defaults["eta_0"],
            log_scale=True,
            latex_label=r"$\eta_0$ [Pa·s]",
        )
        self.create_parameter_control(
            param_frame,
            "m",
            "m [s]",
            3.0e-4,
            6.0,
            self.defaults["m"],
            log_scale=True,
            latex_label=r"$m$ [s]",
        )
        self.create_parameter_control(
            param_frame,
            "n",
            "n [-]",
            0.60,
            0.70,
            self.defaults["n"],
            latex_label=r"$n$ [-]",
        )
        self.create_parameter_control(
            param_frame,
            "tan_delta",
            "tan(δ) [-]",
            0.5,
            35.0,
            self.defaults["tan_delta"],
            log_scale=True,
            latex_label=r"$\tan(\delta)$ [-]",
        )
        self.create_parameter_control(
            param_frame,
            "D_ph",
            "D_Ph\n[mmol-Ph/mL]",
            0,
            1.5e-2,
            self.defaults["D_ph"],
            latex_label=r"$D_{Ph}$"
            + "\n"
            + r"$[\mathrm{mmol}$"
            + "-"
            + r"$\mathrm{Ph/mL}]$",
        )

        # Separator
        sep2 = ctk.CTkFrame(self.left_panel, height=2, fg_color="gray50")
        sep2.pack(fill="x", padx=10, pady=10)

        # Temperature control
        temp_frame = ctk.CTkFrame(self.left_panel)
        temp_frame.pack(fill="x", padx=10, pady=5)

        temp_label = ctk.CTkLabel(
            temp_frame,
            text="Generation Settings",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        temp_label.pack(pady=(5, 10))

        temp_row = ctk.CTkFrame(temp_frame)
        temp_row.pack(fill="x", padx=5, pady=2)
        temp_latex_img = render_latex_label(r"Temperature ($T$)", fontsize=11)
        ctk.CTkLabel(
            temp_row, text="", image=temp_latex_img, width=150, anchor="center"
        ).pack(side="left", padx=5)
        self.temp_latex_img = (
            temp_latex_img  # Keep reference to prevent garbage collection
        )
        self.temp_slider = ctk.CTkSlider(temp_row, from_=0, to=1, number_of_steps=100)
        self.temp_slider.set(0.7)
        self.temp_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.temp_value = ctk.CTkLabel(
            temp_row, text="0.70", width=50, font=ctk.CTkFont(size=14)
        )
        self.temp_value.pack(side="left", padx=5)
        self.temp_slider.configure(command=self.on_temp_slider_change)

        # On-the-fly generation toggle
        self.on_the_fly_var = ctk.BooleanVar(value=False)
        self.on_the_fly_check = ctk.CTkCheckBox(
            temp_frame,
            text="On-the-fly Generation",
            variable=self.on_the_fly_var,
            command=self.on_toggle_on_the_fly,
            font=ctk.CTkFont(size=16),
        )
        self.on_the_fly_check.pack(pady=10)

        # Buttons
        btn_frame = ctk.CTkFrame(self.left_panel)
        btn_frame.pack(fill="x", padx=10, pady=20)

        self.generate_btn = ctk.CTkButton(
            btn_frame,
            text="Generate Image",
            command=self.generate_image,
            font=ctk.CTkFont(size=20, weight="bold"),
            height=50,
        )
        self.generate_btn.pack(fill="x", padx=5, pady=5)

        # bind enter key to generate image
        self.bind("<Return>", lambda event: self.generate_image())

        self.save_btn = ctk.CTkButton(
            btn_frame,
            text="Save Image",
            command=self.save_image,
            state="disabled",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.save_btn.pack(fill="x", padx=5, pady=5)

        # Model status
        self.status_frame = ctk.CTkFrame(self.left_panel)
        self.status_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            self.status_frame,
            text="Model Status:",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=5)
        self.cvae_status = ctk.CTkLabel(
            self.status_frame, text="β-CVAE: Not loaded", text_color="orange"
        )
        self.cvae_status.pack(anchor="w", padx=10)
        self.svr_status = ctk.CTkLabel(
            self.status_frame, text="SVR: Not loaded", text_color="orange"
        )
        self.svr_status.pack(anchor="w", padx=10)
        self.cnn_status = ctk.CTkLabel(
            self.status_frame, text="CNN: Not loaded", text_color="orange"
        )
        self.cnn_status.pack(anchor="w", padx=10)

        # GUI Version block
        self.version_frame = ctk.CTkFrame(self.left_panel)
        self.version_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(
            self.version_frame,
            text="GUI Version:",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=5)
        ctk.CTkLabel(
            self.version_frame,
            text="v1.1.0",
            font=ctk.CTkFont(size=14),
            text_color="gray40",
        ).pack(anchor="w", padx=10)

    def on_alg_slider_change(self, value):
        self.alg_entry.delete(0, "end")
        self.alg_entry.insert(0, f"{value:.2f}")
        self.apply_concentrations_from_sliders()

    def on_ha_slider_change(self, value):
        self.ha_entry.delete(0, "end")
        self.ha_entry.insert(0, f"{value:.2f}")
        self.apply_concentrations_from_sliders()

    def on_alg_entry_confirm(self, event=None):
        try:
            val = float(self.alg_entry.get())
            val = np.clip(val, 0, 5.0)
            self.alg_entry.delete(0, "end")
            self.alg_entry.insert(0, f"{val:.2f}")
            self.alg_slider.set(val)
            self.apply_concentrations_from_sliders()
        except ValueError:
            pass

    def on_ha_entry_confirm(self, event=None):
        try:
            val = float(self.ha_entry.get())
            val = np.clip(val, 0, 3.0)
            self.ha_entry.delete(0, "end")
            self.ha_entry.insert(0, f"{val:.2f}")
            self.ha_slider.set(val)
            self.apply_concentrations_from_sliders()
        except ValueError:
            pass

    def on_temp_slider_change(self, value):
        self.temp_value.configure(text=f"{value:.2f}")
        if self.on_the_fly_var.get():
            self.trigger_generation()

    def on_toggle_on_the_fly(self):
        if self.on_the_fly_var.get():
            self.generate_btn.configure(state="disabled", text="Auto-Generating...")
            self.trigger_generation()
        else:
            self.generate_btn.configure(state="normal", text="Generate Image")

    def apply_concentrations_from_sliders(self):
        if self.updating_from_rheology:
            return

        self.updating_from_concentration = True

        try:
            alg_conc = float(self.alg_entry.get()) if self.alg_entry.get() else 0
            ha_conc = float(self.ha_entry.get()) if self.ha_entry.get() else 0
        except ValueError:
            self.updating_from_concentration = False
            return

        # Handle edge case: both at 0
        if alg_conc < 0.1 and ha_conc < 0.1:
            alg_conc = 0.25
            ha_conc = 0.25
            self.alg_entry.delete(0, "end")
            self.alg_entry.insert(0, "0.25")
            self.ha_entry.delete(0, "end")
            self.ha_entry.insert(0, "0.25")
            self.alg_slider.set(0.25)
            self.ha_slider.set(0.25)

        # Predict rheology using SVR
        if self.svr_models is not None:
            rheology = self.predict_rheology(alg_conc, ha_conc)
            self.set_parameter("eta_0", rheology["eta0"], from_concentration=True)
            self.set_parameter("m", rheology["m"], from_concentration=True)
            self.set_parameter("n", rheology["n"], from_concentration=True)
            self.set_parameter(
                "tan_delta", rheology["tan_delta"], from_concentration=True
            )

        # Calculate D_Ph
        D_ph = self.calculate_ph_content(alg_conc, ha_conc)
        self.set_parameter("D_ph", D_ph, from_concentration=True)

        self.concentration_mode = True
        self.mode_label.configure(
            text="Mode: SVR Rheological Predictors", text_color="green"
        )

        self.updating_from_concentration = False

        if self.on_the_fly_var.get():
            self.trigger_generation()

    def load_different_model(self):
        filepath = filedialog.askopenfilename(
            title="Select β-CVAE Model",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")],
            initialdir="models/beta_CVAE",
        )
        if filepath:
            self.current_model_path = filepath
            model_name = Path(filepath).name
            self.model_name_label.configure(text=model_name)

            # Check if "grid" is in the model name
            if "grid" in model_name.lower():
                self.cnn_enabled = True
                self.load_cnn_model()
                self.cnn_bottom_right_title.configure(
                    text="CNN Predicted Metrics (Grid Model)"
                )
            else:
                self.cnn_enabled = False
                self.cnn_model = None
                self.cnn_status.configure(
                    text="CNN: Disabled (Non-grid Model)", text_color="gray"
                )
                self.cnn_bottom_right_title.configure(
                    text="CNN Metrics Disabled (Non-grid Model)"
                )
                for key in self.metric_labels:
                    self.metric_labels[key].configure(text="N/A", text_color="gray")

            self.load_cvae_model(filepath)

    def create_parameter_control(
        self,
        parent,
        name,
        label,
        min_val,
        max_val,
        default,
        log_scale=False,
        latex_label=None,
    ):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=3)

        if latex_label:
            latex_img = render_latex_label(latex_label, fontsize=11)
            lbl = ctk.CTkLabel(
                frame, text="", image=latex_img, width=150, anchor="center"
            )
        else:
            lbl = ctk.CTkLabel(
                frame, text=label, width=100, font=ctk.CTkFont(size=14), anchor="center"
            )
        lbl.pack(side="left", padx=5)

        entry = ctk.CTkEntry(frame, width=100, font=ctk.CTkFont(size=14))
        entry.pack(side="left", padx=5)
        entry.insert(0, f"{default:.4g}")
        entry.bind("<KeyRelease>", lambda e, n=name: self.on_param_entry_change(n))
        entry.bind("<Return>", lambda e, n=name: self.on_param_entry_confirm(n))

        if log_scale:
            slider_min = np.log10(max(min_val, 1e-10))
            slider_max = np.log10(max_val)
            slider_default = np.log10(max(default, 1e-10))
        else:
            slider_min, slider_max, slider_default = min_val, max_val, default

        slider = ctk.CTkSlider(frame, from_=slider_min, to=slider_max, width=150)
        slider.set(slider_default)
        slider.pack(side="left", fill="x", expand=True, padx=5)

        def on_slider_change(val, n=name, e=entry, ls=log_scale):
            actual_val = 10**val if ls else val
            e.delete(0, "end")
            e.insert(0, f"{actual_val:.4g}")
            self.on_rheology_slider_change(n)

        slider.configure(command=on_slider_change)

        self.param_widgets[name] = {
            "entry": entry,
            "slider": slider,
            "log_scale": log_scale,
            "min": min_val,
            "max": max_val,
        }

    def on_rheology_slider_change(self, name):
        """Called when a rheological parameter slider is moved manually."""
        if self.updating_from_concentration:
            return
        if name != "intensity":
            if name == "D_ph":
                self.reset_concentration_sliders(reset_entries=False)
            else:
                self.reset_concentration_sliders()
        if self.on_the_fly_var.get():
            self.trigger_generation()

    def on_param_entry_change(self, name):
        if self.updating_from_concentration:
            return
        if name != "intensity":
            if name == "D_ph":
                self.reset_concentration_sliders(reset_entries=False)
            else:
                self.reset_concentration_sliders()

    def on_param_entry_confirm(self, name):
        widget = self.param_widgets[name]
        try:
            val = float(widget["entry"].get())
            val = np.clip(val, widget["min"], widget["max"])
            widget["entry"].delete(0, "end")
            widget["entry"].insert(0, f"{val:.4g}")
            if widget["log_scale"]:
                widget["slider"].set(np.log10(max(val, 1e-10)))
            else:
                widget["slider"].set(val)
            if name != "intensity":
                if name == "D_ph":
                    self.reset_concentration_sliders(reset_entries=False)
                else:
                    self.reset_concentration_sliders()
            if self.on_the_fly_var.get():
                self.trigger_generation()
        except ValueError:
            pass

    def reset_concentration_sliders(self, reset_entries=True):
        """Reset concentration sliders to 0 when rheological parameters are changed manually."""
        if self.updating_from_concentration:
            return
        self.updating_from_rheology = True
        self.alg_slider.set(0)
        self.ha_slider.set(0)
        if reset_entries:
            self.alg_entry.delete(0, "end")
            self.alg_entry.insert(0, "0")
            self.ha_entry.delete(0, "end")
            self.ha_entry.insert(0, "0")
        self.concentration_mode = False
        self.mode_label.configure(
            text="Mode: Manual Parameter Sweeping", text_color="#dc8e00"
        )
        self.updating_from_rheology = False

    def trigger_generation(self):
        """Trigger image generation with debounce."""
        if hasattr(self, "_generation_pending"):
            self.after_cancel(self._generation_pending)
        self._generation_pending = self.after(200, self.generate_image)

    def switch_to_parameter_mode(self):
        if self.concentration_mode:
            self.concentration_mode = False
            self.mode_label.configure(
                text="Mode: Manual Parameter Sweeping", text_color="orange"
            )
            self.reset_concentration_sliders()

    def apply_concentrations(self):
        # This method is no longer needed as sliders auto-apply
        pass

    def calculate_ph_content(self, alg_percent, ha_percent):
        """Calculate Ph content from concentrations using constants."""
        alg_conc = (alg_percent / 100) * MMOL_PER_GRAM_ALG
        ha_conc = (ha_percent / 100) * MMOL_PER_GRAM_HA
        return alg_conc + ha_conc

    def predict_rheology(self, alg_conc, ha_conc):
        """Predict rheological parameters using SVR models."""
        X = np.array([[alg_conc, ha_conc]])
        X_scaled = self.svr_scaler_x.transform(X)

        results = {}
        for name in ["eta0", "m", "n", "tan_delta"]:
            y_pred_scaled = self.svr_models[name].predict(X_scaled)
            y_pred_transformed = self.svr_scalers_y[name].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )[0, 0]
            if name != "n":
                results[name] = np.exp(y_pred_transformed)
            else:
                results[name] = y_pred_transformed

        return results

    def set_parameter(self, name, value, from_concentration=False):
        """Set a parameter value. If from_concentration=True, don't reset concentration sliders."""
        widget = self.param_widgets[name]
        widget["entry"].delete(0, "end")
        widget["entry"].insert(0, f"{value:.4g}")
        if widget["log_scale"]:
            widget["slider"].set(np.log10(max(value, 1e-10)))
        else:
            widget["slider"].set(value)

    def get_parameter(self, name):
        try:
            return float(self.param_widgets[name]["entry"].get())
        except ValueError:
            return self.defaults.get(name, 0)

    def setup_right_panel(self):
        # Image display frame
        self.image_frame = ctk.CTkFrame(self.right_panel)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="Awaiting Image Generation",
            font=ctk.CTkFont(size=20),
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # CNN metrics frame
        self.metrics_frame = ctk.CTkFrame(self.right_panel)
        self.metrics_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.cnn_bottom_right_title = ctk.CTkLabel(
            self.metrics_frame,
            text="CNN Predicted Metrics (Grid Model)",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.cnn_bottom_right_title.pack(anchor="center", padx=10, pady=5)

        self.metrics_container = ctk.CTkFrame(self.metrics_frame)
        self.metrics_container.pack(fill="x", padx=10, pady=5)

        self.metric_labels = {}
        metrics_info = [
            ("area", r"Area [mm$^2$]"),
            ("width", r"Filament Width [mm]"),
            ("hd", r"HD [-]"),
            ("dice", r"DSC [-]"),
            ("complete", r"Completeness [%]"),
        ]
        self.metric_latex_imgs = {}
        for i, (key, latex_label) in enumerate(metrics_info):
            frame = ctk.CTkFrame(self.metrics_container)
            frame.pack(side="left", padx=10, pady=5, expand=True)
            latex_img = render_latex_label(latex_label, fontsize=12)
            self.metric_latex_imgs[key] = latex_img
            ctk.CTkLabel(frame, text="", image=latex_img).pack()
            self.metric_labels[key] = ctk.CTkLabel(
                frame, text="--", font=ctk.CTkFont(size=18, weight="bold")
            )
            self.metric_labels[key].pack()

    def on_resize(self, event):
        # Debounce resize events
        current_size = (self.winfo_width(), self.winfo_height())
        if current_size == self._last_size:
            return
        self._last_size = current_size

        # Update image if we have one
        if self.generated_image is not None:
            self.after(100, self.update_image_display)

    def update_image_display(self):
        if self.generated_image is None:
            return

        # Get available space in the image frame
        frame_width = self.image_frame.winfo_width() - 20
        frame_height = self.image_frame.winfo_height() - 20

        if frame_width < 100 or frame_height < 100:
            return

        # Calculate size maintaining aspect ratio (square image)
        size = min(frame_width, frame_height, 800)
        size = max(size, 128)

        # Resize image
        pil_img_resized = self.generated_image.resize((size, size), Image.LANCZOS)

        # Display
        ctk_image = ctk.CTkImage(
            light_image=pil_img_resized, dark_image=pil_img_resized, size=(size, size)
        )
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image

    def load_models(self):
        self.load_cvae_model()
        self.load_svr_models()
        self.load_cnn_model()

    def load_cvae_model(self, model_path=None):
        if model_path is None:
            model_path = Path(self.current_model_path)
        else:
            model_path = Path(model_path)

        scaler_json_path = model_path.parent / "scaler_statistics.json"
        scaler_pkl_path = model_path.parent / "condition_scaler.pkl"

        try:
            from CVAE_hyperparameters import LATENT_DIM, CONDITION_DIM, BETA

            self.model = BetaCVAE(
                latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM, beta=BETA
            )
        except ImportError:
            self.model = BetaCVAE(latent_dim=32, condition_dim=6, beta=5.0)

        if model_path.exists():
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            # Infer latent dim from checkpoint
            if "encoder.fc_mu.bias" in checkpoint:
                inferred_latent_dim = checkpoint["encoder.fc_mu.bias"].shape[0]
                self.model = BetaCVAE(latent_dim=inferred_latent_dim, condition_dim=6)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.cvae_status.configure(text=f"β-CVAE: Loaded", text_color="green")
        else:
            self.cvae_status.configure(text="β-CVAE: File not found", text_color="red")

        # Load scaler - prefer JSON
        if scaler_json_path.exists():
            with open(scaler_json_path, "r") as f:
                stats = json.load(f)
                from sklearn.preprocessing import StandardScaler

                self.cvae_scaler = StandardScaler()
                self.cvae_scaler.mean_ = np.array(stats["mean"])
                self.cvae_scaler.scale_ = np.array(stats["std"])
                self.cvae_scaler.var_ = self.cvae_scaler.scale_**2
                self.cvae_scaler.n_features_in_ = len(stats["mean"])
        elif scaler_pkl_path.exists():
            try:
                with open(scaler_pkl_path, "rb") as f:
                    self.cvae_scaler = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load β-CVAE scaler: {e}")

    def load_svr_models(self):
        svr_dir = Path("models/SVR")
        if not svr_dir.exists():
            self.svr_status.configure(text="SVR: Directory not found", text_color="red")
            return

        try:
            self.svr_models = {}
            self.svr_scalers_y = {}

            for name in ["eta0", "m", "n", "tan_delta"]:
                with open(svr_dir / f"svr_{name}_model.pkl", "rb") as f:
                    self.svr_models[name] = pickle.load(f)
                with open(svr_dir / f"scaler_Y_{name}.pkl", "rb") as f:
                    self.svr_scalers_y[name] = pickle.load(f)

            with open(svr_dir / "scaler_X.pkl", "rb") as f:
                self.svr_scaler_x = pickle.load(f)

            self.svr_status.configure(text="SVR: Loaded", text_color="green")
        except Exception as e:
            self.svr_status.configure(
                text=f"SVR: Error - {str(e)[:30]}", text_color="red"
            )
            self.svr_models = None

    def load_cnn_model(self):
        cnn_dir = Path("models/CNN")
        if not cnn_dir.exists():
            self.cnn_status.configure(text="CNN: Directory not found", text_color="red")
            return

        try:
            import tensorflow as tf

            self.cnn_model = tf.keras.models.load_model(
                cnn_dir / "multioutput_model.keras"
            )
            self.cnn_scalers = joblib.load(cnn_dir / "multioutput_scalers.pkl")
            self.regression_targets = [
                "construct_area_mm2",
                "average_filament_width_mm",
                "average_hd",
                "dice_coefficient",
            ]
            self.cnn_status.configure(text="CNN: Loaded", text_color="green")
        except Exception as e:
            self.cnn_status.configure(
                text=f"CNN: Error - {str(e)[:30]}", text_color="red"
            )
            self.cnn_model = None

    def predict_cnn_metrics(self, cvae_image):
        """Predict metrics using CNN model."""
        if self.cnn_model is None or not self.cnn_enabled:
            return None

        # Convert to grayscale and prepare for CNN
        img = cvae_image.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        cnn_input = gray.reshape(1, 128, 128, 1)

        predictions = self.cnn_model.predict(cnn_input, verbose=0)
        pred_reg, pred_cls = predictions

        result = {}
        for j, target in enumerate(self.regression_targets):
            result[target] = self.cnn_scalers[target].inverse_transform(
                [[pred_reg[0, j]]]
            )[0, 0]
        result["completeness_prob"] = pred_cls[0, 0]

        return result

    def generate_image(self):
        if self.model is None:
            messagebox.showerror("Error", "β-CVAE model not loaded.")
            return

        # Get parameters
        intensity = self.get_parameter("intensity") / 100.0
        eta_0 = self.get_parameter("eta_0")
        m = self.get_parameter("m")
        n = self.get_parameter("n")
        tan_delta = self.get_parameter("tan_delta")
        D_ph = self.get_parameter("D_ph")
        temperature = self.temp_slider.get()

        # Create condition vector
        condition = np.array(
            [
                intensity,
                np.log10(max(eta_0, 1e-10)),
                np.log10(max(m, 1e-10)),
                n,
                np.log1p(D_ph),
                tan_delta,
            ]
        )

        # Scale condition
        if self.cvae_scaler is not None:
            condition = self.cvae_scaler.transform(condition.reshape(1, -1))[0]

        condition_tensor = (
            torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Generate image
        with torch.no_grad():
            generated = self.model.generate(condition_tensor, temperature=temperature)
            self.generated_image_array = generated[0].cpu().numpy()
            img_array = self.generated_image_array.transpose(1, 2, 0)
            img_array = np.clip(img_array, 0, 1)
            img_array_uint8 = (img_array * 255).astype(np.uint8)

        # Store original PIL image
        self.generated_image = Image.fromarray(img_array_uint8)

        # Update display with current window size
        self.update_image_display()

        # Predict CNN metrics if enabled
        if self.cnn_enabled and self.cnn_model is not None:
            metrics = self.predict_cnn_metrics(self.generated_image_array)
            if metrics:
                self.metric_labels["area"].configure(
                    text=f"{metrics['construct_area_mm2']:.2f}", text_color="black"
                )
                self.metric_labels["width"].configure(
                    text=f"{metrics['average_filament_width_mm']:.2f}",
                    text_color="black",
                )
                self.metric_labels["hd"].configure(
                    text=f"{metrics['average_hd']:.2f}", text_color="black"
                )
                self.metric_labels["dice"].configure(
                    text=f"{metrics['dice_coefficient']:.2f}", text_color="black"
                )
                prob = metrics["completeness_prob"]
                color = "green" if prob > 0.75 else "orange" if prob > 0.5 else "red"
                self.metric_labels["complete"].configure(
                    text=f"{prob:.1%}".replace("%", ""), text_color=color
                )
        else:
            for key in self.metric_labels:
                self.metric_labels[key].configure(text="N/A", text_color="gray")

        self.save_btn.configure(state="normal")

    def save_image(self):
        if self.generated_image is None:
            return

        # Create GUI_save folder if it doesn't exist
        save_dir = Path("GUI_save")
        save_dir.mkdir(exist_ok=True)

        # Get current parameters for filename
        intensity = self.get_parameter("intensity")
        eta_0 = self.get_parameter("eta_0")
        m = self.get_parameter("m")
        n = self.get_parameter("n")
        tan_delta = self.get_parameter("tan_delta")
        D_ph = self.get_parameter("D_ph")
        temperature = self.temp_slider.get()

        # Get concentration values
        try:
            alg_conc = float(self.alg_entry.get()) if self.alg_entry.get() else 0
            ha_conc = float(self.ha_entry.get()) if self.ha_entry.get() else 0
        except ValueError:
            alg_conc, ha_conc = 0, 0

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename with conditions
        if self.concentration_mode and (alg_conc > 0 or ha_conc > 0):
            filename = f"ALG{alg_conc:.1f}_HA{ha_conc:.1f}_I{intensity:.0f}_T{temperature:.2f}_{timestamp}.png"
        else:
            filename = f"eta{eta_0:.2e}_m{m:.2e}_n{n:.2f}_tan{tan_delta:.1f}_Dph{D_ph:.4f}_I{intensity:.0f}_T{temperature:.2f}_{timestamp}.png"

        filepath = save_dir / filename

        # Save at 800x800
        img_800 = self.generated_image.resize((800, 800), Image.LANCZOS)
        img_800.save(filepath)

        # Show brief confirmation
        self.save_btn.configure(text=f"Saved in Folder 'GUI_save'", state="normal")
        self.after(1500, lambda: self.save_btn.configure(text="Save Image"))


if __name__ == "__main__":
    app = CVAEGeneratorApp()
    app.mainloop()

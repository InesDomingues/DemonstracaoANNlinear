"""Interactive educational single-neuron demonstration: hypertension vs Age & BMI.

The records and their labels are entirely synthetic. This is not a diagnostic,
screening, or clinical risk-prediction tool; actual hypertension assessment requires
blood-pressure measurements and appropriate clinical evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_SYNTHETIC_SAMPLES = 850
DEFAULT_SEED = 42
MAX_TRAINING_SAMPLES = 400

# Pedagogical setting: smaller values give more clearly separable classes.
# 0.22 = clear boundary with a few ambiguous cases; 0.60 = more overlap.
SYNTHETIC_BOUNDARY_NOISE = 0.22
SYNTHETIC_DATA_VERSION = "clear-separation-age-bmi-v1"

GREEN = "#24845c"
RED = "#d14b5b"
BLUE = "#205e9f"


def sigmoid(z):
    """Stable sigmoid function for a single logistic neuron."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height in centimetres and weight in kilograms."""
    height_m = float(height_cm) / 100.0
    if height_m <= 0 or float(weight_kg) <= 0:
        raise ValueError("Height and weight must be positive.")
    return float(weight_kg) / (height_m ** 2)


def generate_synthetic_dataset(n=DEFAULT_SYNTHETIC_SAMPLES, seed=DEFAULT_SEED):
    """Generate clearly separable *fictional* hypertension cases.

    Age, height and weight are simulated; BMI is calculated from height and
    weight. Labels are assigned using an INVENTED approximately linear boundary
    in Age/BMI space, with limited noise near that boundary. This makes it easy
    to watch a single neuron learn. The rule is NOT derived from patients,
    epidemiology, or diagnostic thresholds.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 86, size=n)
    desired_bmi = np.clip(rng.normal(25.8 + 0.052 * (age - 45), 4.6), 17.0, 43.0)

    # BMI is derived from the same measurements requested from participants.
    height_cm = np.round(np.clip(rng.normal(170, 9, size=n), 145, 200), 1)
    weight_kg = np.round(desired_bmi * (height_cm / 100.0) ** 2, 1)
    bmi = np.round(weight_kg / (height_cm / 100.0) ** 2, 1)

    # Made-up classification rule, deliberately easy for a linear neuron.
    # Older age and higher BMI move a point towards the synthetic 'Yes' class.
    # The small noise term allows a few challenging cases near the boundary.
    synthetic_score = 0.085 * (age - 50) + 0.23 * (bmi - 27) - 0.10
    noisy_score = synthetic_score + rng.normal(0, SYNTHETIC_BOUNDARY_NOISE, size=n)
    hypertension = np.where(noisy_score >= 0, "Yes", "No")

    return pd.DataFrame(
        {
            "Age": age.astype(int),
            "Height (cm)": height_cm,
            "Weight (kg)": weight_kg,
            "BMI": bmi,
            "Hypertension": hypertension,
            "Source": "Synthetic",
        }
    )


def prepare_training_data(df, max_samples=MAX_TRAINING_SAMPLES):
    """Use Age/BMI as features and Hypertension (Yes=1, No=0) as target."""
    valid = df[["Age", "BMI", "Hypertension"]].dropna().copy()
    valid = valid[valid["Hypertension"].isin(["Yes", "No"])]
    if len(valid) < 2 or valid["Hypertension"].nunique() < 2:
        raise ValueError("At least two observations, including both classes, are required.")
    if len(valid) > max_samples:
        valid = valid.sample(n=max_samples, random_state=42)

    x_raw = valid[["Age", "BMI"]].to_numpy(dtype=float)
    y = (valid["Hypertension"] == "Yes").astype(float).to_numpy().reshape(-1, 1)
    mu = x_raw.mean(axis=0, keepdims=True)
    sigma = x_raw.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    x = (x_raw - mu) / sigma
    return x, y, x_raw, mu, sigma


def original_scale_weights(state):
    """Return coefficients (Age in years, BMI in kg/m², intercept)."""
    w_norm = state["w"].reshape(-1)
    mu = state["mu"].reshape(-1)
    sd = state["sigma"].reshape(-1)
    w_orig = w_norm / sd
    intercept = float(state["b"][0, 0] - np.sum(w_orig * mu))
    return float(w_orig[0]), float(w_orig[1]), intercept


def update_metrics(state):
    z = state["X"] @ state["w"] + state["b"]
    p = sigmoid(z)
    y = state["y"]
    eps = 1e-9
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    acc = np.mean((p >= 0.5) == (y >= 0.5))
    state["loss"] = float(loss)
    state["accuracy"] = float(acc)
    return state


def initialize_network(df, seed=42):
    rng = np.random.default_rng(seed)
    x, y, x_raw, mu, sigma = prepare_training_data(df)
    state = {
        "X": x,
        "y": y,
        "X_raw": x_raw,
        "mu": mu,
        "sigma": sigma,
        "w": rng.normal(0, 0.5, size=(2, 1)),
        "b": np.zeros((1, 1)),
        "epoch": 0,
        "age_min": 18.0,
        "age_max": 85.0,
        "bmi_min": 17.0,
        "bmi_max": 43.0,
    }
    return update_metrics(state)


def train_iteration(state, learning_rate=0.3):
    """One full-batch gradient descent step for binary cross-entropy."""
    x, y = state["X"], state["y"]
    p = sigmoid(x @ state["w"] + state["b"])
    residual = p - y
    gradient_w = (x.T @ residual) / len(y)
    gradient_b = np.mean(residual, keepdims=True)
    state["w"] = state["w"] - learning_rate * gradient_w
    state["b"] = state["b"] - learning_rate * gradient_b
    state["epoch"] += 1
    return update_metrics(state)


def draw_decision_boundary(ax, state):
    w_age, w_bmi, b = original_scale_weights(state)
    ages = np.linspace(state["age_min"], state["age_max"], 300)
    if abs(w_bmi) > 1e-10:
        bmis = -(w_age * ages + b) / w_bmi
        ax.plot(ages, bmis, "--", color="black", lw=2, label="Decision boundary (p = 0.5)")
    elif abs(w_age) > 1e-10:
        ax.axvline(-b / w_age, color="black", ls="--", lw=2, label="Decision boundary (p = 0.5)")


def draw_scatter(df, state=None, compact=False):
    fig, ax = plt.subplots(figsize=(6.0, 4.5) if compact else (6.7, 4.5))
    for label, color in [("No", GREEN), ("Yes", RED)]:
        subset = df[(df["Hypertension"] == label) & (df["Source"] == "Synthetic")]
        if not subset.empty:
            ax.scatter(subset["Age"], subset["BMI"], s=13, alpha=0.36, color=color,
                       edgecolors="none", label=f"Hypertension: {label}")
        hypothetical = df[(df["Hypertension"] == label) & (df["Source"] == "Hypothetical")]
        if not hypothetical.empty:
            ax.scatter(hypothetical["Age"], hypothetical["BMI"], s=85, alpha=1.0,
                       color=color, edgecolors="#202833", linewidths=1.2, marker="o",
                       label=f"Hypothetical: {label}")
    if state is not None:
        draw_decision_boundary(ax, state)
    ax.set_xlim(18, 85)
    ax.set_ylim(17, 43)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("BMI (kg/m²)")
    ax.set_title("Synthetic hypertension data" if state is None else
                 f"Training step {state['epoch']}  |  Loss {state['loss']:.3f}  |  Training accuracy {state['accuracy']:.1%}")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def draw_neuron(state):
    """Draw the two inputs and their changing linear coefficients."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        ((0.12, 0.79), "Age", "#d9efff"),
        ((0.12, 0.55), "BMI", "#d9efff"),
        ((0.12, 0.31), "Bias", "#e1e4e8"),
        ((0.51, 0.55), "z", "#f7e6a2"),
        ((0.72, 0.55), "σ", "#f5dcbd"),
        ((0.91, 0.55), "P(Yes)", "#ebd9f7"),
    ]
    w_age, w_bmi, b = original_scale_weights(state)

    def edge(start, end, value=None, dashed=False):
        color = "#238a5c" if value is not None and value >= 0 else "#c84954" if value is not None else "#303844"
        lw = 1.8 if value is None else 1.2 + min(abs(value), 2) * 1.5
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": lw,
                                                            "color": color, "linestyle": "--" if dashed else "-"})

    edge(nodes[0][0], nodes[3][0], w_age)
    edge(nodes[1][0], nodes[3][0], w_bmi)
    edge(nodes[2][0], nodes[3][0], b, dashed=True)
    edge(nodes[3][0], nodes[4][0])
    edge(nodes[4][0], nodes[5][0])

    for (x, y), label, color in nodes:
        circ = plt.Circle((x, y), 0.062, fc=color, ec="#374151", zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=10 if label != "P(Yes)" else 8,
                weight="bold", zorder=4)

    for y, value, prefix in [(0.75, w_age, "w_age"), (0.60, w_bmi, "w_bmi"), (0.35, b, "b")]:
        ax.text(0.30, y, f"{prefix} = {value:+.4f}", ha="center", fontsize=9,
                color="#238a5c" if value >= 0 else "#c84954")
    ax.text(0.67, 0.39, "z = w_age × Age + w_bmi × BMI + b", ha="center", fontsize=10)
    ax.text(0.72, 0.29, r"$\sigma(z)=\frac{1}{1+e^{-z}}$", ha="center", fontsize=13)
    ax.text(0.5, 0.08, "Output: estimated probability of Hypertension = Yes",
            ha="center", fontsize=10, weight="bold")
    ax.set_title("Single-neuron model · weights change during training", fontsize=12)
    fig.tight_layout()
    return fig


# -----------------------------
# App and state
# -----------------------------
st.set_page_config(page_title="Age + BMI → Hypertension | ANN demonstration", page_icon="🧠", layout="wide")
st.title("🧠 How does a neural network learn?")
st.write("Enter **Age, Height and Weight**. The app calculates **BMI = Weight (kg) / Height (m)²** automatically. The neuron then uses only **Age and BMI** as its two input features to predict a simulated label: **Hypertension (Yes / No)**.")
st.warning("**Educational simulation — not a clinical prediction tool.** All initial records and their hypertension labels are invented. Age and BMI alone cannot diagnose hypertension; diagnosis requires blood-pressure assessment.")

if ("demo_dataset" not in st.session_state or
        st.session_state.get("synthetic_data_version") != SYNTHETIC_DATA_VERSION):
    # Refresh older datasets automatically when switching to this code version.
    st.session_state.demo_dataset = generate_synthetic_dataset()
    st.session_state.dataset_version = st.session_state.get("dataset_version", -1) + 1
    st.session_state.synthetic_data_version = SYNTHETIC_DATA_VERSION


df = st.session_state.demo_dataset

with st.expander("How were the synthetic data generated?", expanded=False):
    st.markdown("""
    **Age** is simulated between 18 and 85 years. Fictional **Height (cm)** and **Weight (kg)** are generated, and **BMI** is calculated from them using the same formula as for the participant inputs. A deliberately invented **linear classification rule** assigns the binary **Hypertension** label. A small amount of randomness is added close to the decision boundary, making the synthetic groups much easier to separate than real clinical data.

    This rule is **not based on a clinical cohort, guidelines or European population prevalence**, and the demonstration must not be used for personal health assessment.

    The logistic neuron learns its own parameters from the simulated examples rather than receiving the original generating formula.
    """)

st.subheader("1 · Explore the dataset")
col_add, col_summary = st.columns([1, 1])
with col_add:
    st.markdown("**Add a fictional case (optional)**")
    with st.form("add_case", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        age_new = c1.number_input("Age (years)", min_value=18, max_value=85, value=50, step=1)
        height_new = c2.number_input("Height (cm)", min_value=140.0, max_value=210.0, value=170.0, step=1.0)
        weight_new = c3.number_input("Weight (kg)", min_value=40.0, max_value=180.0, value=78.0, step=0.5)
        label_new = st.selectbox("Fictional label: Hypertension", ["No", "Yes"])
        st.caption("BMI is calculated automatically after submission; only Age and BMI are used as model features.")
        submitted = st.form_submit_button("Add hypothetical case")

    if submitted:
        bmi_new = round(calculate_bmi(height_new, weight_new), 1)
        if not 17.0 <= bmi_new <= 43.0:
            st.error(f"Calculated BMI: {bmi_new:.1f} kg/m². Please choose hypothetical measurements resulting in BMI between 17 and 43 for this simulation.")
        else:
            new_row = pd.DataFrame([{"Age": int(age_new),
                                     "Height (cm)": float(height_new),
                                     "Weight (kg)": float(weight_new),
                                     "BMI": bmi_new,
                                     "Hypertension": label_new, "Source": "Hypothetical"}])
            st.session_state.demo_dataset = pd.concat([df, new_row], ignore_index=True)
            st.session_state.dataset_version += 1
            st.success(f"BMI = {float(weight_new):.1f} / ({float(height_new)/100:.2f})² = {bmi_new:.1f} kg/m². Case added. Training will restart with the updated dataset.")
            st.rerun()

with col_summary:
    total = len(df)
    yes = int((df["Hypertension"] == "Yes").sum())
    no = total - yes
    h = int((df["Source"] == "Hypothetical").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Cases", f"{total:,}")
    c2.metric("Yes (simulated)", f"{yes}")
    c3.metric("No (simulated)", f"{no}")
    st.caption(f"{h} hypothetical cases entered in this browser session. No files are written or uploaded.")
    if st.button("Reset fictional dataset"):
        st.session_state.demo_dataset = generate_synthetic_dataset()
        st.session_state.dataset_version += 1
        st.rerun()

with st.expander("View example rows", expanded=False):
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

fig = draw_scatter(df)
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.divider()
st.subheader("2 · Train the model step by step")
st.write("One sigmoid neuron computes **z = w_age × Age + w_bmi × BMI + b** and outputs **P(Hypertension = Yes) = sigmoid(z)**. Age and BMI are standardised for training; displayed coefficients are converted back to original units. The **decision boundary is a straight line**.")

if ("model_state" not in st.session_state or
        st.session_state.get("model_dataset_version") != st.session_state.dataset_version):
    st.session_state.model_state = initialize_network(df)
    st.session_state.model_dataset_version = st.session_state.dataset_version

col_plot, col_network = st.columns([1, 1])
with col_plot:
    state = st.session_state.model_state
    fig = draw_scatter(pd.DataFrame({"Age": state["X_raw"][:, 0],
                                     "BMI": state["X_raw"][:, 1],
                                     "Hypertension": np.where(state["y"].ravel() == 1, "Yes", "No"),
                                     "Source": "Synthetic"}), state=state, compact=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Train +1 step", use_container_width=True):
            st.session_state.model_state = train_iteration(st.session_state.model_state)
            st.rerun()
    with b2:
        if st.button("Train +10 steps", use_container_width=True):
            for _ in range(10):
                st.session_state.model_state = train_iteration(st.session_state.model_state)
            st.rerun()
    with b3:
        if st.button("Restart weights", use_container_width=True):
            random_seed = int(np.random.default_rng().integers(1, 10000000))
            st.session_state.model_state = initialize_network(df, seed=random_seed)
            st.rerun()

with col_network:
    fig = draw_neuron(st.session_state.model_state)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Green arrows: positive coefficients. Red arrows: negative coefficients. The numbers update as the neuron trains.")

st.info("**Training accuracy** is measured on the same examples used to update the neuron. It is not a validated estimate of performance on new patients. The purpose is to visualise learning and the decision boundary.")

st.divider()
st.subheader("3 · Enter Age, Height and Weight for a hypothetical case")
ca, cb, cc = st.columns(3)
with ca:
    test_age = st.slider("Age (years)", min_value=18, max_value=85, value=55, step=1)
with cb:
    test_height = st.slider("Height (cm)", min_value=140, max_value=210, value=170, step=1)
with cc:
    test_weight = st.slider("Weight (kg)", min_value=40.0, max_value=180.0, value=84.0, step=0.5)

test_bmi = round(calculate_bmi(test_height, test_weight), 1)
st.info(f"**Automatic feature engineering:** BMI = {test_weight:.1f} kg / ({test_height / 100:.2f} m)² = **{test_bmi:.1f} kg/m²**. The model receives **Age = {test_age}** and **BMI = {test_bmi:.1f}**; it does not receive Height or Weight as separate inputs.")
if not 17.0 <= test_bmi <= 43.0:
    st.warning("Calculated BMI is outside the range of the synthetic training examples (approximately 17–43 kg/m²). The model is extrapolating; this demonstration must not be interpreted clinically.")

state = st.session_state.model_state
w_age, w_bmi, b = original_scale_weights(state)
z = w_age * test_age + w_bmi * test_bmi + b
p_yes = float(sigmoid(z))
p_no = 1 - p_yes

st.code(
    f"z = ({w_age:.5f} × {test_age}) + ({w_bmi:.5f} × {test_bmi:.1f}) + ({b:.5f})\n"
    f"z = {z:.5f}\n"
    f"P(Hypertension = Yes) = sigmoid(z) = {p_yes:.3f}\n"
    f"P(Hypertension = No) = 1 − P(Yes) = {p_no:.3f}",
    language="text",
)
st.metric("Simulated model output: P(Yes)", f"{p_yes:.1%}")
st.markdown(f"**Predicted class (threshold 0.50): {'Yes' if p_yes >= 0.5 else 'No'}**")
st.caption("Illustrative model probability only — not a diagnosis and not an individual's calibrated risk of hypertension. The participant inputs Height and Weight are used only to calculate the BMI feature.")

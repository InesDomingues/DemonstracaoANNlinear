from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Configuration
# =============================
DATASET_FILE = Path("adultos_portugal_streamlit.csv")
DEFAULT_SYNTHETIC_SAMPLES = 5000
DEFAULT_SEED = 42

# =============================
# Helper functions
# =============================
    
def criar_figura_rede_ann_linear(estado):
    """Desenha a rede neuronal linear que está a ser treinada.

    Ordem visual dos neurónios de entrada:
    1. peso_kg
    2. altura_cm
    3. bias

    Nota:
    O modelo usa X = ["altura_cm", "peso_kg"].
    Por isso:
    - w_altura = estado["w"][0, 0]
    - w_peso = estado["w"][1, 0]
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pos_peso = (0.12, 0.78)
    pos_altura = (0.12, 0.55)
    pos_bias = (0.12, 0.32)
    pos_neuronio = (0.50, 0.55)
    pos_sigmoid = (0.70, 0.55)
    pos_saida = (0.90, 0.55)

    w_altura, w_peso, b = obter_pesos_em_unidades_originais(estado)

    def espessura(valor):
        return 1 + 3 * min(abs(valor), 2.0) / 2.0

    def cor_peso(valor):
        return "green" if valor >= 0 else "red"

    # Ligações das entradas para o neurónio linear z
    ax.annotate(
        "",
        xy=pos_neuronio,
        xytext=pos_peso,
        arrowprops=dict(
            arrowstyle="->",
            lw=espessura(w_peso),
            color=cor_peso(w_peso),
        ),
    )

    ax.annotate(
        "",
        xy=pos_neuronio,
        xytext=pos_altura,
        arrowprops=dict(
            arrowstyle="->",
            lw=espessura(w_altura),
            color=cor_peso(w_altura),
        ),
    )

    ax.annotate(
        "",
        xy=pos_neuronio,
        xytext=pos_bias,
        arrowprops=dict(
            arrowstyle="->",
            lw=espessura(b),
            color=cor_peso(b),
            linestyle="dashed",
        ),
    )

    # Ligação z -> sigmoid -> saída
    ax.annotate(
        "",
        xy=pos_sigmoid,
        xytext=pos_neuronio,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color="black",
        ),
    )

    ax.annotate(
        "",
        xy=pos_saida,
        xytext=pos_sigmoid,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color="black",
        ),
    )

    # Nós da rede
    nos = [
        (pos_peso, "peso", "lightblue"),
        (pos_altura, "altura", "lightblue"),
        (pos_bias, "bias", "lightgray"),
        (pos_neuronio, "z", "khaki"),
        (pos_sigmoid, "σ", "wheat"),
        (pos_saida, "y_prob", "plum"),
    ]

    for (x, y), label, color in nos:
        circle = plt.Circle(
            (x, y),
            0.06,
            color=color,
            ec="black",
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            zorder=4,
        )

    # Etiquetas dos pesos
    ax.text(
        0.31,
        0.72,
        f"w_peso = {w_peso:.4f}",
        fontsize=9,
        color=cor_peso(w_peso),
        ha="center",
    )

    ax.text(
        0.31,
        0.57,
        f"w_altura = {w_altura:.4f}",
        fontsize=9,
        color=cor_peso(w_altura),
        ha="center",
    )

    ax.text(
        0.31,
        0.36,
        f"b = {b:.4f}",
        fontsize=9,
        color=cor_peso(b),
        ha="center",
    )

    # Equações
    ax.text(
        0.7,
        0.4,
        "z = w_altura·altura + w_peso·peso + b",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.text(
        0.75,
        0.3,
        r"$\sigma(z)=\frac{1}{1+e^{-z}}$",
        ha="center",
        va="center",
        fontsize=13,
    )

    ax.text(
        0.5,
        0.0,
        "Saída: probabilidade estimada de ser Masculino",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.set_title(
        f"Rede neuronal linear em treino",
        fontsize=12,
    )

    return fig
    
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def preparar_dados_demo_ann_linear(df: pd.DataFrame, max_amostras: int = 400):
    """Prepara dados para ANN linear.

    A rede treina com variáveis normalizadas,
    mas depois convertemos os pesos para unidades originais.
    """
    df_demo = df[["altura_cm", "peso_kg", "sexo"]].dropna().copy()

    if len(df_demo) > max_amostras:
        df_demo = df_demo.sample(n=max_amostras, random_state=42)

    X_raw = df_demo[["altura_cm", "peso_kg"]].to_numpy(dtype=float)
    y = (df_demo["sexo"] == "Masculino").astype(float).to_numpy().reshape(-1, 1)

    mu = X_raw.mean(axis=0, keepdims=True)
    sigma = X_raw.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0

    X = (X_raw - mu) / sigma

    return X, y, X_raw, mu, sigma


def obter_pesos_em_unidades_originais(estado):
    """Converte os pesos aprendidos em variáveis normalizadas
    para uma equação nas unidades originais.

    Modelo interno:
        z = w1 * altura_norm + w2 * peso_norm + b

    Equivalente em unidades originais:
        z = w_altura * altura_cm + w_peso * peso_kg + b_original
    """
    w_norm = estado["w"]
    b_norm = float(estado["b"][0, 0])

    mu_altura = estado["mu"][0, 0]
    mu_peso = estado["mu"][0, 1]

    sd_altura = estado["sigma"][0, 0]
    sd_peso = estado["sigma"][0, 1]

    w_altura = float(w_norm[0, 0] / sd_altura)
    w_peso = float(w_norm[1, 0] / sd_peso)

    b_original = (
        b_norm
        - float(w_norm[0, 0] * mu_altura / sd_altura)
        - float(w_norm[1, 0] * mu_peso / sd_peso)
    )

    return w_altura, w_peso, b_original


def calcular_metricas_demo_ann_linear(estado):
    """Calcula loss e accuracy."""
    X = estado["X"]
    y = estado["y"]

    z = X @ estado["w"] + estado["b"]
    y_prob = sigmoid(z)

    eps = 1e-8
    loss = -np.mean(
        y * np.log(y_prob + eps) + (1 - y) * np.log(1 - y_prob + eps)
    )

    y_pred = (y_prob >= 0.5).astype(int)
    acc = np.mean(y_pred == y)

    estado["loss"] = float(loss)
    estado["acc"] = float(acc)
    estado["y_prob"] = y_prob

    return estado


def inicializar_demo_ann_linear(df: pd.DataFrame, seed: int = 42):
    """Inicializa a ANN linear."""
    rng = np.random.default_rng(seed)

    X, y, X_raw, mu, sigma = preparar_dados_demo_ann_linear(df)

    x_min = float(X_raw[:, 0].min() - 5)
    x_max = float(X_raw[:, 0].max() + 5)
    y_min = float(X_raw[:, 1].min() - 8)
    y_max = float(X_raw[:, 1].max() + 8)

    estado = {
        "X": X,
        "y": y,
        "X_raw": X_raw,
        "mu": mu,
        "sigma": sigma,
        "w": rng.normal(0, 0.5, size=(2, 1)),
        "b": np.zeros((1, 1)),
        "epoch": 0,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    estado = calcular_metricas_demo_ann_linear(estado)
    return estado


def treinar_uma_iteracao_demo_ann_linear(estado, lr: float = 0.3):
    """Treina a ANN linear durante uma iteração."""
    X = estado["X"]
    y = estado["y"]
    w = estado["w"]
    b = estado["b"]

    m = X.shape[0]

    z = X @ w + b
    y_prob = sigmoid(z)

    dz = y_prob - y
    dw = (X.T @ dz) / m
    db = np.mean(dz, keepdims=True)

    estado["w"] = w - lr * dw
    estado["b"] = b - lr * db
    estado["epoch"] += 1

    estado = calcular_metricas_demo_ann_linear(estado)
    return estado

def criar_figura_demo_ann_linear(estado):
    """Cria figura com dados e recta de decisão da ANN linear.

    A recta é desenhada nas unidades originais:
    z = w_altura * altura_cm + w_peso * peso_kg + b

    Internamente, o treino usa dados normalizados, mas os pesos são convertidos
    para unidades originais para facilitar a interpretação.
    """
    X_raw = estado["X_raw"]
    y = estado["y"].ravel()

    fig, ax = plt.subplots(figsize=(5, 4))

    femininos = X_raw[y == 0]
    masculinos = X_raw[y == 1]

    if len(femininos) > 0:
        ax.scatter(
            femininos[:, 0],
            femininos[:, 1],
            c="red",
            label="Feminino",
            alpha=0.6,
            s=18,
        )

    if len(masculinos) > 0:
        ax.scatter(
            masculinos[:, 0],
            masculinos[:, 1],
            c="blue",
            label="Masculino",
            alpha=0.6,
            s=18,
        )

    # Limites fixos para o zoom não mudar
    x_min = estado["x_min"]
    x_max = estado["x_max"]
    y_min = estado["y_min"]
    y_max = estado["y_max"]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Recta de decisão convertida para unidades originais
    w_altura, w_peso, b_original = obter_pesos_em_unidades_originais(estado)

    x_vals = np.linspace(x_min, x_max, 200)

    if abs(w_peso) > 1e-12:
        y_vals = -(w_altura * x_vals + b_original) / w_peso

        ax.plot(
            x_vals,
            y_vals,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Recta de decisão",
        )
    else:
        if abs(w_altura) > 1e-12:
            x_vertical = -b_original / w_altura
            ax.axvline(
                x=x_vertical,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Recta de decisão",
            )

    ax.set_xlabel("Altura (cm)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title(
        f"ANN linear | iteração = {estado['epoch']} | "
        f"loss = {estado['loss']:.3f} | acc = {estado['acc']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig
    
def classificar_imc(valor: float) -> str:
    """Classifica o IMC de acordo com as categorias habituais da OMS."""
    if valor < 18.5:
        return "Baixo peso"
    if valor < 25:
        return "Peso normal"
    if valor < 30:
        return "Excesso de peso"
    return "Obesidade"

def gerar_dataset_sintetico_adultos_portugal(
    n: int = DEFAULT_SYNTHETIC_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Gera um dataset sintético simples de adultos em Portugal.

    Nota: os dados são simulados e servem apenas para fins pedagógicos.
    """
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 86, size=n)

    sexo = rng.choice(
        ["Feminino", "Masculino"],
        size=n,
        p=[0.52, 0.48],
    )

    altura_cm = np.zeros(n)

    for i in range(n):
        if sexo[i] == "Feminino":
            altura_cm[i] = rng.normal(loc=162, scale=3.0)
        else:
            altura_cm[i] = rng.normal(loc=174, scale=3.5)

    altura_cm = np.clip(altura_cm, 140, 205)

    categoria_imc = rng.choice(
        ["Baixo peso", "Peso normal", "Excesso de peso", "Obesidade"],
        size=n,
        p=[0.02, 0.38, 0.38, 0.22],
    )

    imc = np.zeros(n)

    for i in range(n):
        if categoria_imc[i] == "Baixo peso":
            imc[i] = rng.normal(loc=17.8, scale=0.7)
        elif categoria_imc[i] == "Peso normal":
            imc[i] = rng.normal(loc=22.5, scale=1.7)
        elif categoria_imc[i] == "Excesso de peso":
            imc[i] = rng.normal(loc=27.2, scale=1.4)
        else:
            imc[i] = rng.normal(loc=33.0, scale=3.2)

    imc = np.clip(imc, 15, 50)

    altura_m = altura_cm / 100
    peso_kg = imc * (altura_m**2)
    peso_kg = peso_kg + rng.normal(loc=0, scale=1.5, size=n)
    peso_kg = np.clip(peso_kg, 40, 180)

    imc_final = peso_kg / (altura_m**2)
    categoria_imc_final = [classificar_imc(x) for x in imc_final]

    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "idade": idade,
            "sexo": sexo,
            "altura_cm": np.round(altura_cm, 1),
            "peso_kg": np.round(peso_kg, 1),
            "imc": np.round(imc_final, 1),
            "categoria_imc": categoria_imc_final,
            "origem": "Sintética",
        }
    )

    return df

def carregar_ou_criar_dataset(n_synthetic_samples: int, seed: int) -> pd.DataFrame:
    """Carrega o CSV existente ou cria um dataset sintético inicial."""
    if DATASET_FILE.exists():
        df = pd.read_csv(DATASET_FILE)

        # Compatibilidade com versões antigas do ficheiro.
        if "origem" not in df.columns:
            df["origem"] = np.where(
                df.index < n_synthetic_samples,
                "Sintética",
                "Real",
            )

        return df

    df = gerar_dataset_sintetico_adultos_portugal(
        n=n_synthetic_samples,
        seed=seed,
    )
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")
    return df

def guardar_dataset(df: pd.DataFrame) -> None:
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")

def criar_scatter_altura_peso(df: pd.DataFrame, distinguir_origem: bool = True, plt_reg: bool = False ):
    """Cria scatter plot: altura no eixo x, peso no eixo y, sexo nas cores.

    Se distinguir_origem=True:
    - dados sintéticos aparecem como pontos pequenos;
    - dados reais aparecem como bolinhas preenchidas.

    Inclui uma linha de regressão linear usando todos os dados.
    """
    cores_sexo = {
        "Feminino": "red",
        "Masculino": "blue",
    }

    fig, ax = plt.subplots(figsize=(4, 3))

    if distinguir_origem:
        for sexo, cor in cores_sexo.items():
            # Dados sintéticos: pontos pequenos
            dados_sinteticos = df[
                (df["sexo"] == sexo) & (df["origem"] == "Sintética")
            ]

            if len(dados_sinteticos) > 0:
                ax.scatter(
                    dados_sinteticos["altura_cm"],
                    dados_sinteticos["peso_kg"],
                    c=cor,
                    marker=".",
                    label=f"{sexo} - Sintética",
                    alpha=0.1,
                    s=1,
                )

            # Dados reais: bolinhas preenchidas
            dados_reais = df[
                (df["sexo"] == sexo) & (df["origem"] == "Real")
            ]

            if len(dados_reais) > 0:
                ax.scatter(
                    dados_reais["altura_cm"],
                    dados_reais["peso_kg"],
                    c=cor,
                    marker="o",
                    label=f"{sexo} - Real",
                    alpha=1,
                    edgecolors="black",
                    linewidths=0.4,
                    s=45,
                )

    else:
        for sexo, cor in cores_sexo.items():
            dados = df[df["sexo"] == sexo]
            ax.scatter(
                dados["altura_cm"],
                dados["peso_kg"],
                c=cor,
                label=sexo,
                alpha=0.6,
                edgecolors="none",
                s=15,
            )

    if plt_reg:
        # Linha de regressão usando todos os dados
        df_reg = df[["altura_cm", "peso_kg"]].dropna()
    
        if len(df_reg) >= 2:
            x = df_reg["altura_cm"].to_numpy()
            y = df_reg["peso_kg"].to_numpy()
    
            coef = np.polyfit(x, y, deg=1)
            linha = np.poly1d(coef)
    
            x_linha = np.linspace(x.min(), x.max(), 100)
            y_linha = linha(x_linha)
    
            ax.plot(
                x_linha,
                y_linha,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Regressão linear",
            )

    ax.set_xlabel("Altura (cm)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Altura vs peso por sexo")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    return fig

# =============================
# Streamlit app
# =============================
st.set_page_config(
    page_title="Exemplo: Dataset de altura e peso",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exemplo: Dataset de altura e peso")

st.write(
    "Esta aplicação permite recolher dados e juntá-los a um dataset sintético inicial."
)

nr_synthetic_samples = DEFAULT_SYNTHETIC_SAMPLES
seed = DEFAULT_SEED

df = carregar_ou_criar_dataset(nr_synthetic_samples, seed)

# Garante ids válidos, caso o utilizador altere manualmente o CSV.
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))

# =============================
# Data collection
# =============================
col_form, col_info = st.columns([1, 1])

with col_form:
    st.subheader("Adicionar nova observação")

    with st.form("student_form"):
        idade = st.number_input(
            "Idade",
            min_value=18,
            max_value=100,
            value=20,
            step=1,
        )

        sexo = st.selectbox(
            "Sexo",
            ["Feminino", "Masculino"],
        )

        altura_cm = st.number_input(
            "Altura (cm)",
            min_value=120.0,
            max_value=220.0,
            value=170.0,
            step=0.1,
        )

        peso_kg = st.number_input(
            "Peso (kg)",
            min_value=35.0,
            max_value=220.0,
            value=70.0,
            step=0.1,
        )

        submitted = st.form_submit_button("Adicionar ao dataset")

    if submitted:
        altura_m = altura_cm / 100
        imc = peso_kg / (altura_m**2)
        categoria_imc = classificar_imc(imc)

        new_id = int(df["id"].max()) + 1 if len(df) > 0 else 1

        new_row = pd.DataFrame(
            [
                {
                    "id": new_id,
                    "idade": int(idade),
                    "sexo": sexo,
                    "altura_cm": round(float(altura_cm), 1),
                    "peso_kg": round(float(peso_kg), 1),
                    "imc": round(float(imc), 1),
                    "categoria_imc": categoria_imc,
                    "origem": "Real",
                }
            ]
        )

        df = pd.concat([df, new_row], ignore_index=True)
        guardar_dataset(df)

        st.success("Observação adicionada com sucesso.")
        st.write(f"IMC: **{imc:.1f}**")
        st.write(f"Categoria de IMC: **{categoria_imc}**")

with col_info:
    st.subheader("Resumo")

    total = len(df)
    n_sinteticas = int((df["origem"] == "Sintética").sum())
    n_reais = int((df["origem"] == "Real").sum())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total", total)
    metric_col2.metric("Sintéticas", n_sinteticas)
    metric_col3.metric("Reais", n_reais)

    st.write("Distribuição por sexo")
    dist_sexo = df["sexo"].value_counts().rename_axis("sexo").reset_index(name="n")
    st.dataframe(dist_sexo, use_container_width=True)

# =============================
# Scatter plot
# =============================
st.subheader("Scatter plot")

fig = criar_scatter_altura_peso(df, distinguir_origem=True)
st.pyplot(fig, use_container_width=False)

# =============================
# Treino de uma rede
# =============================
st.divider()

st.header("Intuição: como é que um modelo aprende?")

st.write(
    "Esta demonstração usa uma rede neuronal muito simples: um único neurónio com ativação sigmoid. "
    "Como só tem uma combinação linear de altura e peso, a fronteira de decisão é sempre uma recta. "
    "A cada clique, os pesos são atualizados e a recta muda de posição."
)

if (
    "demo_ann_linear" not in st.session_state
    or "mu" not in st.session_state.demo_ann_linear
    or "sigma" not in st.session_state.demo_ann_linear
):
    st.session_state.demo_ann_linear = inicializar_demo_ann_linear(df, seed=42)

col_fig1, col_fig2 = st.columns(2)
col_fig1, col_fig2 = st.columns([1, 1])

with col_fig1:
    fig_demo_ann = criar_figura_demo_ann_linear(st.session_state.demo_ann_linear)
    st.pyplot(fig_demo_ann, use_container_width=False)

    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        if st.button("Treinar +1 iteração"):
            st.session_state.demo_ann_linear = treinar_uma_iteracao_demo_ann_linear(
                st.session_state.demo_ann_linear,
                lr=0.3,
            )
            st.rerun()

    with col_b2:
        if st.button("Treinar +10 iterações"):
            for _ in range(10):
                st.session_state.demo_ann_linear = treinar_uma_iteracao_demo_ann_linear(
                    st.session_state.demo_ann_linear,
                    lr=0.3,
                )
            st.rerun()

    with col_b3:
        if st.button("Reiniciar ANN aleatória"):
            nova_seed = np.random.randint(0, 100000)
            st.session_state.demo_ann_linear = inicializar_demo_ann_linear(
                df,
                seed=nova_seed,
            )
            st.rerun()

with col_fig2:
    fig_rede = criar_figura_rede_ann_linear(st.session_state.demo_ann_linear)
    st.pyplot(fig_rede, use_container_width=False)

st.subheader("Testar uma nova pessoa")

col_input_1, col_input_2 = st.columns(2)

with col_input_1:
    altura_nova = st.number_input(
        "Altura da nova pessoa (cm)",
        min_value=120.0,
        max_value=220.0,
        value=160.0,
        step=0.1,
        key="altura_nova_ann",
    )

with col_input_2:
    peso_novo = st.number_input(
        "Peso da nova pessoa (kg)",
        min_value=35.0,
        max_value=220.0,
        value=80.0,
        step=0.1,
        key="peso_novo_ann",
    )

estado = st.session_state.demo_ann_linear

w_altura, w_peso, b = obter_pesos_em_unidades_originais(estado)

z_novo = w_altura * altura_nova + w_peso * peso_novo + b
prob_masculino = float(sigmoid(z_novo))
prob_feminino = 1 - prob_masculino

st.write("Cálculo da rede:")

st.code(
    f"z = ({w_altura:.6f} × {altura_nova:.1f}) "
    f"+ ({w_peso:.6f} × {peso_novo:.1f}) "
    f"+ ({b:.6f})\n"
    f"z = {z_novo:.6f}\n"
    f"P(Masculino) = sigmoid(z) = {prob_masculino:.4f}\n"
    f"P(Feminino) = 1 - P(Masculino) = {prob_feminino:.4f}",
    language="text",
)

if prob_masculino >= 0.5:
    st.success(f"Classe prevista: Masculino ({prob_masculino:.1%})")
else:
    st.success(f"Classe prevista: Feminino ({prob_feminino:.1%})")

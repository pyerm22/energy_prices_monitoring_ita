import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

from prophet.plot import plot_plotly, plot_components_plotly

from epm.models.prophet.forecaster import Forecaster
from epm.scraping_utils.fuel_prices import FuelPrices

st.set_page_config(
    page_title="Prezzi Carburanti",
    page_icon="📈⛽️",
)

st.markdown(
    """
        I dati mostrati in questa pagina rappresentano l'
        **andamento dei prezzi dei carburanti** dal 2005 alla data corrente 
        e sono una rielaborazione di quelli forniti dal 
        [Ministero dell'Ambiente e della Sicurezza Energetica](https://dgsaie.mise.gov.it/open-data).
    """
)

fp = FuelPrices()

@st.cache_data
def get_fuel_prices() -> pd.DataFrame:
    return fp.get_data()

# ✅ IMPORTANT: actually load the data
fuel_prices = get_fuel_prices()

st.write("fuel_prices shape:", fuel_prices.shape)
st.write("fuel_prices head:", fuel_prices.head())

# ✅ Normalize column names ONCE (avoid casing/space issues)
fuel_prices.columns = fuel_prices.columns.astype(str).str.strip().str.upper()

# ✅ Debug helper (keep for now, remove later)
st.caption(f"Available columns in fuel_prices: {list(fuel_prices.columns)}")

# ---- session state defaults ----
if "target_selected" not in st.session_state:
    st.session_state.target_selected = False
if "experiment_name" not in st.session_state:
    st.session_state.experiment_name = None
if "artifact_path" not in st.session_state:
    st.session_state.artifact_path = None
if "train" not in st.session_state:
    st.session_state.train = False
if "predict" not in st.session_state:
    st.session_state.predict = False

def set_experiment() -> None:
    st.session_state["target_selected"] = True

    if st.session_state["target_col"] == "BENZINA":
        st.session_state.experiment_name = "gasoline_model"
        st.session_state.artifact_path = "gasoline_prices_model"
    elif st.session_state["target_col"] == "DIESEL":
        st.session_state.experiment_name = "diesel_model"
        st.session_state.artifact_path = "diesel_prices_model"
    elif st.session_state["target_col"] == "GPL":
        st.session_state.experiment_name = "nlg_model"
        st.session_state.artifact_path = "nlg_prices_model"

def click_train():
    st.session_state.train = True

def click_predict():
    st.session_state.predict = True

st.session_state["model_trained"] = False

@st.cache_resource()
def model_training(train_df: pd.DataFrame) -> Forecaster:
    """
    Returns the fitted forecaster
    """
    horizon = st.session_state["horizon"] * 7
    period = st.session_state["period"] * 7
    initial = round(len(fuel_prices) * 0.75)

    forecaster = Forecaster()
    forecaster.train_model(
        experiment_name=st.session_state["experiment_name"],
        train_df=train_df,
        target_col=st.session_state["target_col"],
        artifact_path=st.session_state["artifact_path"],
        horizon=f"{horizon} days",
        period=f"{period} days",
        initial=f"{initial} days",
    )
    return forecaster

with st.expander(label="Fuel Prices Data"):
    st.dataframe(data=fuel_prices, use_container_width=True)

# -------------------- Main plot area --------------------
if "predictions" not in st.session_state:
    with st.container():
        if not st.session_state["target_selected"]:

            wanted = ["BENZINA", "DIESEL", "GPL"]
            available = [c for c in wanted if c in fuel_prices.columns]
            missing = [c for c in wanted if c not in fuel_prices.columns]

            if missing:
                st.warning(
                    f"Missing expected columns: {missing}. "
                    "I will plot the ones that exist. "
                    "Use the 'Available columns' list above to update mapping if needed."
                )

            if available:
                fig = px.line(
                    data_frame=fuel_prices,
                    y=available,  # ✅ use column names, not Series
                    title="Andamento storico dei prezzi dei carburanti",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write("Puoi selezionare i dati di un carburante per addestrare un modello e effettuare previsioni")
            else:
                st.error(
                    "None of the expected columns (BENZINA, DIESEL, GPL) were found. "
                    "Check the 'Available columns' list above and update the expected names."
                )

        else:
            target_col = str(st.session_state["target_col"]).strip().upper()
            if target_col not in fuel_prices.columns:
                st.error(
                    f"Selected target '{st.session_state['target_col']}' not found. "
                    f"Available columns: {list(fuel_prices.columns)}"
                )
            else:
                fig = px.line(
                    data_frame=fuel_prices,
                    y=target_col,
                    title=f"Andamento storico dei prezzi {target_col} (€/lt)",
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    with st.container():
        if not st.session_state.get("keep_in_sample_forecast", True):
            st.caption(f"Predizione andamento futuro dei prezzi {st.session_state['target_col']}")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state.predictions.tail(st.session_state["n_steps"]),
                xlabel="Data",
                ylabel=f"{st.session_state['target_col']}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Storico + predizione dell'andamento del prezzo selezionato")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state["predictions"],
                xlabel="Data",
                ylabel=f"{st.session_state['target_col']}",
            )
            st.plotly_chart(fig, use_container_width=True)

# -------------------- Target selection --------------------
st.session_state["target_col"] = st.selectbox(
    label="Seleziona il carburante di cui vuoi prevedere l'andamento del prezzo",
    options=("BENZINA", "DIESEL", "GPL"),
    index=None,
    placeholder="Seleziona dati..",
)

if st.session_state.target_col:
    set_experiment()
    col = str(st.session_state["target_col"]).strip().upper()

    if col not in fuel_prices.columns:
        st.error(
            f"Selected fuel '{col}' not found in dataset. "
            f"Available columns: {list(fuel_prices.columns)}"
        )
    else:
        sel_fuel_price = fuel_prices[[col]].copy()
        sel_fuel_price.index.rename("index", inplace=True)

        with st.sidebar:
            st.session_state["horizon"] = st.slider(
                label="Quanto avanti vuoi effettuare le predizioni?",
                help=(
                    "Influenza l'addestramento del modello, indicando il numero di predizioni (settimane) "
                    "che vengono effettuate ad ogni periodo."
                ),
                min_value=1,
                max_value=len(fuel_prices),
                value=4,
                step=1,
            )

            st.session_state["period"] = st.slider(
                label="Quanto spesso intendi utilizzare il modello?",
                help="Indica la frequenza (settimanale) con la quale vengono ri-effettuate le predizioni.",
                min_value=1,
                max_value=len(fuel_prices),
                value=2,
                step=1,
            )

            st.button(label="Addestra il modello!", on_click=click_train)

        if st.session_state["train"]:
            with st.spinner("Addestramento modello in corso.."):
                st.session_state["forecaster"] = model_training(sel_fuel_price)
            st.success("Fatto! Il modello è addestrato e pronto ad effettuare le sue predizioni!")
            st.session_state["model_trained"] = True
        else:
            st.info("Puoi addestrare un algoritmo predittivo su questi dati cliccando sul bottone a sinistra!")

# -------------------- Forecast controls --------------------
if st.session_state.get("model_trained", False):
    with st.expander(label="Configura i parametri per la previsione", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["n_steps"] = st.number_input(
                label="Quante previsioni vuoi far realizzare al modello?",
                min_value=1,
                max_value=len(st.session_state["forecaster"].train_df),
                value=1,
            )
        with col2:
            st.session_state["keep_in_sample_forecast"] = st.checkbox(
                label="Vuoi mantenere i risultati di predizione sul set di addestramento?",
                value=True,
                help="Se selezionato, nel grafico compariranno anche le previsioni effettuate sul dataset di training",
            )

    st.button(label="Effettua la predizione!", on_click=click_predict)

    if st.session_state["predict"]:
        st.session_state["predictions"] = st.session_state["forecaster"].forecast(
            n_steps=st.session_state["n_steps"],
            keep_in_sample_forecast=st.session_state["keep_in_sample_forecast"],
        )

        if st.session_state["keep_in_sample_forecast"]:
            preds = st.session_state["predictions"][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        else:
            preds = st.session_state["predictions"].tail(st.session_state["n_steps"])[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        preds = preds.rename(
            columns={
                "ds": "data",
                "yhat": "predizione",
                "yhat_lower": "predizione_minima",
                "yhat_upper": "predizione_massima",
            }
        )

        st.download_button(
            label="Clicca per scaricare il dato di forecast",
            data=preds.to_csv(index=False),
            file_name=f"forecast_{st.session_state['target_col']}.csv",
        )

        with st.expander(label="Espandi per vedere il dato di forecast"):
            st.dataframe(data=preds, use_container_width=True)

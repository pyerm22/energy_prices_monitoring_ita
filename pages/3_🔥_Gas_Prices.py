import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


from prophet.plot import plot_plotly, plot_components_plotly


from epm.models.prophet.forecaster import Forecaster
from epm.scraping_utils.gas_prices import GasPrices

st.set_page_config(
    page_title="Prezzo del Gas Naturale",
    page_icon="🔥",
)

st.markdown(
    """
    Al contrario del prezzo dell'energia elettrica, quello del gas non è un prezzo unico:  
    infatti, non tutte le compagnie presenti nel business della vendita di gas naturale
    ai consumatori in Italia si riforniscono sulla stessa "piazza".
    I dati qui presentati sono quelli del mercato olandese [TTF](https://www.enel.it/en/supporto/faq/ttf-gas),
    ottenuti tramite l'API di Yahoo Finance ([yfinance](https://pypi.org/project/yfinance/)).  
    Con il tempo, il TTF ha assunto il ruolo di indice di riferimento per il prezzo del gas naturale nel mercato europeo.
    """
)

@st.cache_data()
def get_gas_prices() -> pd.DataFrame:
    gp = GasPrices.get_data()
    return gp

gas_prices = get_gas_prices()

experiment_name = "gas_model" # set the model training experiment name for mlflow
target_col = "GAS NATURALE"
artifact_path = "gas_prices_model"

if 'train' not in st.session_state:
    st.session_state.train = False

def click_train():
    st.session_state.train = True

if "predict" not in st.session_state:
    st.session_state.predict = False

def click_predict():
    st.session_state.predict = True 

st.session_state["model_trained"] = False

with st.sidebar:

    st.session_state["horizon"] = st.slider(
        label="Quanto avanti vuoi effettuare le predizioni?",
        help="Influenza l'addestramento del modello, indicando il numero di predizioni (settimane) che vengono effettuate ad ogni periodo. Idealmente, l'orizzonte dovrebbe essere simile a quello che ci si aspetta per predire nel futuro con il modello adesstrato.",
        min_value=1,
        max_value=len(gas_prices),
        value=4,
        step=1
    )   

    st.session_state["period"] = st.slider(
        label="Quanto spesso intendi utilizzare il modello?", 
        help="Influenza l'addestramento del modello, indicando la frequenza (settimanale) con la quale vengono ri-effettuate le predizioni.",
        min_value = 1,
        max_value = len(gas_prices),
        value=2,
        step=1
    )

    st.button(label="Addestra il modello!", on_click=click_train)

@st.cache_resource()
def model_training() -> Forecaster:
    """
    Returns the fitted forecaster
    """
    horizon = st.session_state["horizon"]*7
    period = st.session_state["period"]*7
    initial = round(len(gas_prices)*0.75) 

    forecaster = Forecaster()

    forecaster.train_model(
        experiment_name=experiment_name,
        train_df=gas_prices,
        target_col=target_col,
        artifact_path=artifact_path,
        horizon=f"{horizon} days",
        period=f"{period} days",
        initial=f"{initial} days"
    )
    return forecaster

with st.expander(label="Dati Gas Naturale (TTF)"):
    st.dataframe(data=gas_prices, use_container_width=True)

if "predictions" not in st.session_state:
    with st.container():
        fig = px.line(
            gas_prices, 
            y='GAS NATURALE', 
            title='Andamento storico dei prezzi del Gas Naturale',
            labels={
                "GAS NATURALE": "Prezzi TTF (€/smc)",
                "index": "Data"
            }
        )
        st.plotly_chart(fig)

else:
    with st.container():
        if not st.session_state["keep_in_sample_forecast"]:
            st.caption("Predizione andamento futuro del prezzo del Gas Naturale")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state.predictions.tail(st.session_state["n_steps"]),
                xlabel="Data",
                ylabel="Prezzi TTF (€/smc)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Storico + predizione dell'andamento del prezzo del Gas Naturale")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state["predictions"],
                xlabel="Data",
                ylabel="Prezzi TTF (€/smc)"
            )
            st.plotly_chart(fig, use_container_width=True)

if st.session_state["train"]:
    with st.spinner("Addestramento modello in corso.."):
        st.session_state["forecaster"] = model_training()
    st.success('Fatto! Il modello è addestrato e pronto ad effettuare le sue predizioni!')
    st.session_state["model_trained"] = True
else: 
    st.info(
        "Puoi addestrare un algoritmo predittivo su questi dati cliccando sul bottone a sinistra!"
    )

if st.session_state["model_trained"]:
    
    with st.expander(label="Configura i parametri per la previsione", expanded=True):
        col1, col2 = st.columns(2)
        # input per la previsione
        with col1:
            st.session_state["n_steps"] = st.number_input(
                label="Quante previsioni vuoi far realizzare al modello?",
                min_value=1,
                max_value=len(st.session_state["forecaster"].train_df),
                value=1
            )
        with col2:
            st.session_state["keep_in_sample_forecast"] = st.checkbox(
                label="Vuoi mantenere i risultati di predizione sul set di addestramento?",
                value=True,
                help="Se selezionato, nel grafico compariranno anche le previsioni che il modello ha effettuato sul dataset di training"
            )

    st.button(label="Effettua la predizione!", on_click=click_predict)
    
    if st.session_state["predict"]:
        st.session_state["predictions"] = st.session_state["forecaster"].forecast(
            n_steps=st.session_state["n_steps"],
            keep_in_sample_forecast=st.session_state["keep_in_sample_forecast"]
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
                "yhat_upper": "predizione_massima"
            }
        )
        st.download_button(
            label="Clicca per scaricare il dato di forecast",
            data=preds.to_csv(),
            file_name="forecast_TTF.csv"
        )
        
        with st.expander(label="Espandi per vedere il dato di forecast"):
            st.dataframe(data=preds)

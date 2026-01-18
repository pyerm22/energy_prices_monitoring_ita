import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prophet.plot import plot_plotly, plot_components_plotly


from epm.models.prophet.forecaster import Forecaster
from epm.scraping_utils.elec_prices import ElectricityPrices

st.set_page_config(
    page_title="Prezzo Unico Nazionale",
    page_icon="📈⚡️",
)

st.markdown(
    """
        Il PUN (acronimo di Prezzo Unico Nazionale) è il prezzo di
        riferimento all'ingrosso dell’energia elettrica che viene acquistata 
        sul mercato della Borsa Elettrica Italiana (**IPEX - Italian Power Exchange**).  

        Il PUN rappresenta, la media pesata nazionale dei prezzi zonali di vendita 
        dell’energia elettrica per ogni ora e per ogni giorno. 
        Il dato nazionale è un importo che viene calcolato sulla media di diversi 
        fattori, e che tiene conto delle quantità e dei prezzi formati nelle diverse
        zone d’Italia e nelle diverse ore della giornata.   

        Il dato proposto nell'app è un'aggregazione dei prezzi orari sulle settimane. 
        La fonte del dato è il [Gestore dei Mercati Elettrici](https://www.mercatoelettrico.org/it/)
    """
)
ep = ElectricityPrices()

@st.cache_data
def get_electricity_prices() -> pd.DataFrame:
    pun_prices = ep.get_data()
    return pun_prices

pun_prices = get_electricity_prices()

experiment_name = "electricity_model" # set the model training experiment name for mlflow
target_col = "PUN"
artifact_path = "electricity_prices_model"

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
        max_value=len(pun_prices),
        value=4,
        step=1
    )   

    st.session_state["period"] = st.slider(
        label="Quanto spesso intendi utilizzare il modello?", 
        help="Influenza l'addestramento del modello, indicando la frequenza (settimanale) con la quale vengono ri-effettuate le predizioni.",
        min_value = 1,
        max_value = len(pun_prices),
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
    initial = round(len(pun_prices)*0.75) 

    forecaster = Forecaster()

    forecaster.train_model(
        experiment_name=experiment_name,
        train_df=pun_prices,
        target_col=target_col,
        artifact_path=artifact_path,
        horizon=f"{horizon} days",
        period=f"{period} days",
        initial=f"{initial} days"
    )
    return forecaster


with st.expander(label='Prezzo Unico Nazionale'):
    st.dataframe(data=pun_prices, use_container_width=True)

if "predictions" not in st.session_state:
    with st.container():
        fig = px.line(
            pun_prices, 
            y='PUN', 
            title="Andamento storico del Prezzo Unico Nazionale dell'Energia",
            labels={
                "PUN": "PUN (€/kWh)",
                "index": "Data"
            }
        )
        st.plotly_chart(fig)

else:
    with st.container():
        if not st.session_state["keep_in_sample_forecast"]:
            st.caption("Predizione andamento futuro del Prezzo Unico Nazionale")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state.predictions.tail(st.session_state["n_steps"]),
                xlabel="Data",
                ylabel="PUN (€/kWh)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Storico + predizione dell'andamento del Prezzo Unico Nazionale")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state["predictions"],
                xlabel="Data",
                ylabel="PUN (€/kWh)"
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
            file_name="forecast_PUN.csv"
        )
        
        with st.expander(label="Espandi per vedere il dato di forecast"):
            st.dataframe(data=preds)


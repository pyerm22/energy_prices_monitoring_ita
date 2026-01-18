import datetime
import pandas as pd 
import plotly.express as px
import streamlit as st
from prophet import Prophet

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
    fuel_prices=fp.get_data()
    return fuel_prices

fuel_prices = get_fuel_prices()

st.write("Columns:", list(fuel_prices.columns))
st.dataframe(fuel_prices.head())

if "target_selected" not in st.session_state:
    st.session_state.target_selected = False
if "experiment_name" not in st.session_state:
    st.session_state.experiment_name = None
if "artifact_path" not in st.session_state:
    st.session_state.artifact_path = None

def set_experiment() -> None:
    """
    Depending on the chosen fuel, sets an experiment for mlflow, dropping the 
    columns that are not of interest for the model and returning a df with only 
    date index and the target col. 
    It also sets to True the status of the selectbox.
    """
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

if 'train' not in st.session_state:
    st.session_state.train = False

def click_train():
    st.session_state.train = True

if "predict" not in st.session_state:
    st.session_state.predict = False

def click_predict():
    st.session_state.predict = True 

st.session_state["model_trained"] = False

@st.cache_resource()
def model_training() -> Forecaster:
    """
    Returns the fitted forecaster
    """
    horizon = st.session_state["horizon"]*7
    period = st.session_state["period"]*7
    initial = round(len(fuel_prices)*0.75) 

    forecaster = Forecaster()

    forecaster.train_model(
        experiment_name=st.session_state["experiment_name"],
        train_df=sel_fuel_price,
        target_col=st.session_state["target_col"],
        artifact_path=st.session_state["artifact_path"],
        horizon=f"{horizon} days",
        period=f"{period} days",
        initial=f"{initial} days"
    )
    return forecaster

with st.expander(label='Fuel Prices Data'):
    st.dataframe(data=fuel_prices, use_container_width=True)

if "predictions" not in st.session_state:
    with st.container():
        if not st.session_state["target_selected"]:
            fig = px.line(
                data_frame=fuel_prices,
                y=[fuel_prices["BENZINA"], fuel_prices["DIESEL"], fuel_prices["GPL"]],
                title="Andamento storico dei prezzi dei carburanti"
            )
            st.plotly_chart(fig)
            st.write("Puoi selezionare i dati di un carburante per addestrare un modello e effettuare previsioni")
        else:
            fig = px.line(
                data_frame=fuel_prices,
                y=st.session_state["target_col"],
                title=f'Andamento storico dei prezzi {st.session_state["target_col"]} (€/lt)'
            )
            st.plotly_chart(fig)
else: 
    with st.container():
        if not st.session_state["keep_in_sample_forecast"]:
            st.caption(f'Predizione andamento futuro dei prezzi {st.session_state["target_col"]}')
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state.predictions.tail(st.session_state["n_steps"]),
                xlabel="Data",
                ylabel=f'{st.session_state["target_col"]}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Storico + predizione dell'andamento del Prezzo Unico Nazionale")
            fig = plot_plotly(
                m=st.session_state["forecaster"].model,
                fcst=st.session_state["predictions"],
                xlabel="Data",
                ylabel=f'{st.session_state["target_col"]}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
st.session_state["target_col"] = st.selectbox(
    label="Seleziona il carburante di cui vuoi prevedere l'andamento del prezzo",
    options=("BENZINA", "DIESEL", "GPL"),
    index=None,
    placeholder="Seleziona dati.."
)

if st.session_state.target_col:
    set_experiment()
    col = st.session_state["target_col"]
    sel_fuel_price = fuel_prices[[col]]
    sel_fuel_price.index.rename("index", inplace=True)

    with st.sidebar:
        st.session_state["horizon"] = st.slider(
            label="Quanto avanti vuoi effettuare le predizioni?",
            help="Influenza l'addestramento del modello, indicando il numero di predizioni (settimane) che vengono effettuate ad ogni periodo. Idealmente, l'orizzonte dovrebbe essere simile a quello che ci si aspetta per predire nel futuro con il modello adesstrato.",
            min_value=1,
            max_value=len(fuel_prices),
            value=4,
            step=1
        )   

        st.session_state["period"] = st.slider(
            label="Quanto spesso intendi utilizzare il modello?", 
            help="Influenza l'addestramento del modello, indicando la frequenza (settimanale) con la quale vengono ri-effettuate le predizioni.",
            min_value = 1,
            max_value = len(fuel_prices),
            value=2,
            step=1
        )

        st.button(label="Addestra il modello!", on_click=click_train)

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
            file_name=f'forecast_{st.session_state["target_col"]}.csv'
        )
        
        with st.expander(label="Espandi per vedere il dato di forecast"):
            st.dataframe(data=preds)

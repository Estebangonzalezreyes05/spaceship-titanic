"""Aplicación Streamlit del proyecto Spaceship Titanic."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from train_model import engineer, make_model, SPENDING

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Spaceship Titanic | ML', page_icon='🚀', layout='wide')

@st.cache_resource(show_spinner='Entrenando modelo con los datos del proyecto...')
def load_model():
    train = pd.read_csv(BASE / 'data' / 'train.csv', dtype={'PassengerId': 'string'})
    test = pd.read_csv(BASE / 'data' / 'test.csv', dtype={'PassengerId': 'string'})
    counts = pd.concat([train['PassengerId'], test['PassengerId']]).str.split('_').str[0].value_counts()
    x = engineer(train, counts)
    y = train['Transported'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    if y.isna().any():
        raise ValueError('Etiquetas Transported inválidas')
    model = make_model(RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                               n_jobs=-1, random_state=42))
    model.fit(x, y)
    return model, counts, len(train)

st.title('🚀 Spaceship Titanic')
st.caption('Machine Learning · Clasificación binaria · Proyecto académico')
page = st.sidebar.radio('Navegación', ['Inicio', 'Predecir pasajero', 'Resultados'])

if page == 'Inicio':
    st.header('¿Quién fue transportado?')
    st.write('La aplicación utiliza características de los pasajeros para estimar si fueron transportados a otra dimensión.')
    st.info('El modelo se entrena con los registros de train.csv recuperados de los PDF del proyecto. No es una predicción real ni una evaluación externa de Kaggle.')
    st.markdown('**Características:** planeta de origen, destino, criosueño, edad, VIP, ubicación de cabina, gastos y tamaño del grupo.')
    st.markdown('**Modelo:** Random Forest con imputación de valores faltantes, codificación categórica y creación de variables.')

elif page == 'Predecir pasajero':
    st.header('Simulador de predicción')
    st.write('Completa el formulario con los datos hipotéticos de un pasajero.')
    with st.form('pasajero'):
        a, b = st.columns(2)
        with a:
            home = st.selectbox('Planeta de origen', ['Earth','Europa','Mars'])
            destination = st.selectbox('Destino', ['TRAPPIST-1e','55 Cancri e','PSO J318.5-22'])
            age = st.number_input('Edad', min_value=0, max_value=120, value=30)
            cryo = st.checkbox('Está en criosueño')
            vip = st.checkbox('Es VIP')
        with b:
            deck = st.selectbox('Cubierta', list('ABCDEFGT'))
            cabin_num = st.number_input('Número de cabina', min_value=0, max_value=5000, value=10)
            side = st.selectbox('Lado de cabina', ['P', 'S'], format_func=lambda x: 'Babor (P)' if x=='P' else 'Estribor (S)')
            group_size = st.number_input('Tamaño del grupo', min_value=1, max_value=50, value=1)
        st.subheader('Gastos del pasajero')
        cols = st.columns(5)
        spends = {}
        for col, ui in zip(SPENDING, cols):
            with ui:
                spends[col] = st.number_input(col, min_value=0.0, max_value=1000000.0, value=0.0, step=10.0)
        submit = st.form_submit_button('Calcular predicción', type='primary')
    if submit:
        try:
            model, counts, n = load_model()
            # El identificador simulado tiene un grupo no observado. Modificamos su
            # tamaño de grupo explícitamente al construir las características.
            data = {'PassengerId':'9999_01', 'HomePlanet':home,
                    'CryoSleep':'TRUE' if cryo else 'FALSE',
                    'Cabin':f'{deck}/{int(cabin_num)}/{side}',
                    'Destination':destination,'Age':int(age),
                    'VIP':'TRUE' if vip else 'FALSE', 'Name':'Pasajero Simulado',**spends}
            x = engineer(pd.DataFrame([data]), counts)
            x.loc[x.index[0], 'GroupSize'] = float(group_size)
            probability = float(model.predict_proba(x)[0,1])
            predicted = probability >= 0.5
            st.success('Predicción: **Transportado**' if predicted else 'Predicción: **No transportado**')
            st.metric('Probabilidad estimada de ser transportado', f'{probability:.1%}')
            st.progress(probability)
            st.caption(f'Modelo reentrenado con {n:,} registros; umbral de decisión 50 %. Resultado orientativo.')
        except (OSError, ValueError) as error:
            st.error(f'No se pudo ejecutar la predicción: {error}')

else:
    st.header('Evaluación del modelo')
    report_path = BASE / 'outputs' / 'metricas.json'
    image_path = BASE / 'outputs' / 'matriz_confusion.png'
    if report_path.exists():
        with report_path.open(encoding='utf-8') as f:
            report = json.load(f)
        st.write(f"Modelo seleccionado en la evaluación: **{report['seleccionado']}**")
        df = pd.DataFrame(report['modelos']).T
        st.dataframe(df.style.format('{:.2%}'), use_container_width=True)
        st.bar_chart(df[['accuracy','f1']])
        st.caption('Son métricas de validación guardadas por train_model.py; no son resultados del formulario en tiempo real.')
    else:
        st.warning('Ejecuta primero `python train_model.py` para generar las métricas.')
    if image_path.exists():
        st.image(str(image_path), caption='Matriz de confusión (validación)', use_container_width=False)

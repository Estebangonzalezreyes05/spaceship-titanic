"""Interfaz institucional de Spaceship Titanic (Streamlit)."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from train_model import engineer, make_model, SPENDING

BASE = Path(__file__).resolve().parent
ASSETS = BASE / 'assets'
OUTPUTS = BASE / 'outputs'
UNAM = ASSETS / 'unam_logo.png'
FES = ASSETS / 'fes_logo.jpg'
BANNER = ASSETS / 'banner_redes.png'

st.set_page_config(page_title='Redes Neuronales | FES Acatlán', page_icon='🎓', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
:root {color-scheme: dark;}
[data-testid="stAppViewContainer"] {background: radial-gradient(circle at 85% 8%, #2e2616 0%, #111112 32%, #090909 76%); color: #f3eee0;}
[data-testid="stHeader"] {background: rgba(9,9,9,.86);}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#131313,#211b10); border-right:1px solid #65502a;}
[data-testid="stSidebar"] * {color: #f1e9d6;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#edd9a1!important;}
.block-container {max-width: 1200px; padding-top:2.4rem; padding-bottom:3rem;}
h1,h2,h3 {color:#e9c66d!important; letter-spacing:.01em;}
p,li,label {color:#f2ead7;}
.gold-line {height:2px; background:linear-gradient(90deg,transparent,#e3bf67,transparent); margin:1.1rem 0 1.8rem;}
.hero {border:1px solid #93743c; background:linear-gradient(140deg,#171715,#292216); border-radius:22px; padding:2rem 1.3rem; text-align:center; box-shadow:0 20px 60px #0008;}
.hero .institution {font-size:clamp(1rem,2vw,1.5rem); letter-spacing:.08em; color:#f4e3b7; font-weight:750;}
.hero .school {font-size:clamp(.95rem,1.6vw,1.2rem); color:#d4bd84; margin-top:.7rem;}
.hero .subject {font-size:1.1rem; color:#f2ebdd; margin-top:.75rem;}
.hero .task {font-size:clamp(1.7rem,3.3vw,2.7rem); font-weight:850; color:#f5cd6e; margin:1rem 0;}
.hero .details {font-size:1rem; line-height:1.9; color:#fff7e7;}
.gold-card {border:1px solid #665431; border-radius:16px; background:#1c1a16; padding:1.3rem; margin:.8rem 0; color:#f3e9d2;}
.gold-card strong {color:#f5cf7c;}
[data-testid="stMetric"] {border:1px solid #64502b; border-radius:12px; padding:1rem; background:#211d16;}
[data-testid="stMetricLabel"] p {color:#e4cd95!important;}
[data-testid="stMetricValue"] {color:#ffe0a0!important;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {background:#cda656!important; color:#18140b!important; border:1px solid #ead08f!important; font-weight:750; border-radius:10px;}
.stButton > button:hover, .stFormSubmitButton > button:hover {border-color:#ffe6a1!important; color:#ffe6a1!important;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stImage"] img {border-radius:14px;}
</style>
''', unsafe_allow_html=True)


def image_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def show_logo(path: Path, name: str):
    if image_valid(path):
        st.image(str(path), use_container_width=True)
    else:
        st.caption(f'Logotipo {name} no disponible: revisa assets/{path.name}')


@st.cache_resource(show_spinner='Preparando el modelo de clasificación...')
def load_model(tipo_modelo):
    train = pd.read_csv(BASE / 'data' / 'train.csv', dtype={'PassengerId':'string'})
    test = pd.read_csv(BASE / 'data' / 'test.csv', dtype={'PassengerId':'string'})
    counts = pd.concat([train['PassengerId'], test['PassengerId']]).str.split('_').str[0].value_counts()
    x = engineer(train, counts)
    y = train['Transported'].astype('string').str.upper().map({'TRUE':1, 'FALSE':0})
    if y.isna().any():
        raise ValueError('La variable objetivo Transported contiene valores inválidos.')
    if tipo_modelo == 'Red neuronal (MLP)':
        estimator = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                  solver='adam', alpha=0.001, batch_size=128,
                                  max_iter=250, early_stopping=True,
                                  n_iter_no_change=15, random_state=42)
    else:
        estimator = RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                            n_jobs=-1, random_state=42)
    model = make_model(estimator)
    model.fit(x, y.astype(int))
    return model, counts, len(train)


st.sidebar.markdown('## ✦ FES Acatlán')
st.sidebar.caption('UNAM · Sistemas Inteligentes')
page = st.sidebar.radio('Navegación', ['Portada', 'Predicción', 'Resultados', 'Acerca del proyecto'])
st.sidebar.divider()
st.sidebar.markdown('**Tarea:** REDES NEURONALES')
st.sidebar.markdown('**Alumno:** González Reyes Esteban')
st.sidebar.markdown('**Profesor:** Mendoza Villar José Ricardo')

if page == 'Portada':
    left, center, right = st.columns([1, 3.5, 1], vertical_alignment='center')
    with left:
        show_logo(UNAM, 'UNAM')
    with center:
        st.markdown('''<div class="hero"><div class="institution">UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO</div><div class="school">FACULTAD DE ESTUDIOS SUPERIORES ACATLÁN</div><div class="subject">Sistemas Inteligentes</div><div class="gold-line"></div><div class="task">REDES NEURONALES</div><div class="details"><strong>Profesor:</strong> Mendoza Villar José Ricardo<br><strong>Alumno:</strong> González Reyes Esteban</div></div>''', unsafe_allow_html=True)
    with right:
        show_logo(FES, 'FES Acatlán')
    st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)
    if image_valid(BANNER):
        b1, b2, b3 = st.columns([1, 2.4, 1])
        with b2:
            st.image(str(BANNER), use_container_width=True)
    st.markdown('## Presentación del proyecto')
    st.markdown('''<div class="gold-card">Aplicación interactiva basada en el conjunto de datos <strong>Spaceship Titanic</strong>. Permite introducir características de un pasajero y estimar si fue transportado a otra dimensión. Incluye métricas de evaluación y una matriz de confusión.<p>Se implementan dos algoritmos: <strong>Random Forest</strong> y una <strong>red neuronal MLP</strong> con capas ocultas de 64 y 32 neuronas.</p></div>''', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a: st.metric('Modelos', 'Random Forest / MLP')
    with b: st.metric('Objetivo', 'Transported')
    with c: st.metric('Tipo de problema', 'Binario')

elif page == 'Predicción':
    st.title('Predicción de pasajeros')
    st.markdown('Introduce un perfil hipotético y selecciona el algoritmo para calcular la predicción.')
    tipo_modelo = st.selectbox('Modelo para predecir', ['Red neuronal (MLP)', 'Random Forest'])
    with st.form('passenger_form'):
        a,b = st.columns(2)
        with a:
            home = st.selectbox('Planeta de origen', ['Earth','Europa','Mars'])
            destination = st.selectbox('Destino', ['TRAPPIST-1e','55 Cancri e','PSO J318.5-22'])
            age = st.number_input('Edad', min_value=0, max_value=120, value=30)
            cryo = st.checkbox('En criosueño')
            vip = st.checkbox('Servicio VIP')
        with b:
            deck = st.selectbox('Cubierta', list('ABCDEFGT'))
            cabin_num = st.number_input('Número de cabina', min_value=0, max_value=5000, value=10)
            side = st.selectbox('Lado de cabina', ['P','S'], format_func=lambda x: 'Babor (P)' if x == 'P' else 'Estribor (S)')
            group_size = st.number_input('Tamaño del grupo', min_value=1, max_value=50, value=1)
        st.markdown('### Consumo de servicios')
        cols = st.columns(5)
        spends = {}
        for name, col in zip(SPENDING, cols):
            with col:
                spends[name] = st.number_input(name, min_value=0.0, max_value=1000000.0, value=0.0, step=10.0)
        submitted = st.form_submit_button('Calcular predicción', type='primary')
    if submitted:
        try:
            model, counts, n = load_model(tipo_modelo)
            row = {'PassengerId':'9999_01','HomePlanet':home,'CryoSleep':'TRUE' if cryo else 'FALSE',
                   'Cabin':f'{deck}/{int(cabin_num)}/{side}','Destination':destination,
                   'Age':int(age),'VIP':'TRUE' if vip else 'FALSE','Name':'Pasajero Simulado', **spends}
            xnew = engineer(pd.DataFrame([row]), counts)
            xnew.loc[xnew.index[0], 'GroupSize'] = float(group_size)
            probability = float(model.predict_proba(xnew)[0,1])
            st.success('Resultado estimado: TRANSPORTADO' if probability >= .5 else 'Resultado estimado: NO TRANSPORTADO')
            st.metric('Probabilidad estimada de transporte', f'{probability:.1%}')
            st.progress(probability)
            st.caption(f'Algoritmo: {tipo_modelo}. Modelo entrenado con {n:,} registros. La salida es una estimación, no una certeza.')
        except (OSError, ValueError, KeyError) as err:
            st.error(f'No fue posible realizar la predicción: {err}')

elif page == 'Resultados':
    st.title('Evaluación de los modelos')
    metrics_file = OUTPUTS / 'metricas.json'
    confusion_file = OUTPUTS / 'matriz_confusion.png'
    if metrics_file.exists():
        with metrics_file.open(encoding='utf-8') as fh:
            report = json.load(fh)
        df = pd.DataFrame(report['modelos']).T
        st.markdown(f"**Modelo seleccionado durante la validación:** `{report['seleccionado']}`")
        st.dataframe(df.style.format('{:.2%}'), use_container_width=True)
        st.bar_chart(df[['accuracy','f1']])
        st.caption('Métricas almacenadas por train_model.py; corresponden a la validación interna.')
    else:
        st.warning('Ejecuta `python train_model.py` para generar metricas.json.')
    if image_valid(confusion_file):
        st.subheader('Matriz de confusión')
        st.image(str(confusion_file), caption='Resultados de validación', width=650)
    submission_file = OUTPUTS / 'submission.csv'
    if submission_file.exists():
        st.subheader('Predicciones para el conjunto de prueba')
        result = pd.read_csv(submission_file)
        st.dataframe(result.head(20), use_container_width=True)
        st.download_button('Descargar submission.csv', submission_file.read_bytes(), file_name='submission.csv', mime='text/csv')

else:
    st.title('Acerca del proyecto')
    st.markdown('''<div class="gold-card"><strong>Institución:</strong> Universidad Nacional Autónoma de México — F.E.S. Acatlán<br><strong>Asignatura:</strong> Sistemas Inteligentes<br><strong>Tarea:</strong> REDES NEURONALES<br><strong>Profesor:</strong> Mendoza Villar José Ricardo<br><strong>Alumno:</strong> González Reyes Esteban</div>''', unsafe_allow_html=True)
    st.markdown('''La aplicación utiliza Python, Streamlit y Scikit-learn para entrenar y comparar Random Forest y una red neuronal MLP de dos capas ocultas (64 y 32 neuronas), con predicciones y métricas de validación.''')

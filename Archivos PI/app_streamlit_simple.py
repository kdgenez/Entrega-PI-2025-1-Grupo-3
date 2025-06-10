import streamlit as st
import pandas as pd
from joblib import load

# === CONFIGURACIÓN ===
MODELO_PATH = "mejor_modelo_entrenado.joblib"
NOMBRE_SALIDA = "resultado_con_predicciones.csv"

# === Cargar modelo entrenado ===
@st.cache_resource
def cargar_modelo():
    return load(MODELO_PATH)

def descategorizar(x):
    if x == 1:
        return 'Anomalia'
    elif x == 0:
        return 'Normal'
    else:
        return 'No Revisado'

modelo = cargar_modelo()

# === Interfaz ===
st.title("🧠 Predicción desde archivo CSV")

archivo_csv = st.file_uploader("📂 Sube tu archivo CSV", type="csv")

if archivo_csv is not None:
    try:
        df = pd.read_csv(archivo_csv, encoding='utf-8')
        predicciones = modelo.predict(df)
        df['Pred'] = predicciones
        df['Predicciones'] = df['Pred'].apply(descategorizar)
        df.drop('Pred', axis=1, inplace=True)

        st.success("✅ Predicciones generadas con éxito.")
        st.download_button(
            "📥 Descargar resultados",
            data=df.to_csv(index=False, encoding='utf-8'),
            file_name=NOMBRE_SALIDA,
            mime="text/csv"
        )
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
else:
    st.info("Sube un archivo CSV para iniciar.")

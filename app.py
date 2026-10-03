import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# Crear pestañas para entrada manual o carga de archivos
tab1, tab2 = st.tabs(["Entrada Manual", "Subir Archivo"])

with tab1:
    st.header("Datos de Entrada Individual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder, key="felder_manual")
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01, key="examen_manual")

    # Crear DataFrame temporal con los datos
    df_input = pd.DataFrame({
        'Felder': [felder_input],
        'Examen_admisión': [examen_input]
    })

    if st.button("Realizar Predicción Individual", key="btn_manual"):
        try:
            df_procesado = df_input.copy()
            one_hot_transformer = joblib.load('one_hot_columns.joblib')

            if isinstance(one_hot_transformer, list):
                si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
                for col_name in si_columnas_one_hot:
                    valor_esperado = col_name.replace('Felder_', '')
                    df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
            else:
                df_encoded = pd.get_dummies(df_procesado[['Felder']])
                df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
                si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

            df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

            if isinstance(one_hot_transformer, list):
                for col in si_columnas_one_hot:
                    if col not in df_procesado.columns:
                        df_procesado[col] = 0.0

            scaler = joblib.load('min_max_scaler.joblib')
            df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
            df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

            columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
            df_procesado = df_procesado[columnas_ordenadas]

            st.subheader("Datos Procesados")
            st.dataframe(df_procesado)

            model = joblib.load('bagging_optimizado.joblib')
            prediccion = model.predict(df_procesado)
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Predicción por Lote (Archivo)")
    st.write("Sube un archivo de Excel (.xlsx) o CSV que contenga las columnas `Felder` y `Examen_admisión`.")
    
    uploaded_file = st.file_uploader("Selecciona un archivo", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            
            st.subheader("Vista previa de los datos subidos")
            st.dataframe(df_uploaded.head())
            
            if 'Felder' in df_uploaded.columns and 'Examen_admisión' in df_uploaded.columns:
                if st.button("Procesar y Predecir Archivo", key="btn_lote"):
                    df_procesado = df_uploaded.copy()
                    one_hot_transformer = joblib.load('one_hot_columns.joblib')

                    if isinstance(one_hot_transformer, list):
                        si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
                        for col_name in si_columnas_one_hot:
                            valor_esperado = col_name.replace('Felder_', '')
                            df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
                    else:
                        df_encoded = pd.get_dummies(df_procesado[['Felder']])
                        df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
                        si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

                    df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

                    if isinstance(one_hot_transformer, list):
                        for col in si_columnas_one_hot:
                            if col not in df_procesado.columns:
                                df_procesado[col] = 0.0

                    scaler = joblib.load('min_max_scaler.joblib')
                    df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
                    
                    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
                    df_final_pred = df_procesado[columnas_ordenadas]

                    model = joblib.load('bagging_optimizado.joblib')
                    predicciones = model.predict(df_final_pred)
                    
                    df_resultados = df_uploaded.copy()
                    df_resultados['Prediccion_Nota_Final'] = predicciones
                    
                    st.subheader("Resultados de la Predicción")
                    st.dataframe(df_resultados)
                    
                    # Permitir descargar los resultados
                    csv_data = df_resultados.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Resultados en CSV",
                        data=csv_data,
                        file_name="predicciones_resultados.csv",
                        mime="text/csv"
                    )
            else:
                st.error("El archivo debe contener las columnas 'Felder' y 'Examen_admisión'.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

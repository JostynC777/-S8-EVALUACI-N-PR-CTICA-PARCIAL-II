import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import modelos

# Configuración de la página
st.set_page_config(page_title="Proyecto IA - Examen Final", layout="wide")

st.title("🧠 Proyecto: Algoritmos de Machine Learning desde Cero")
st.markdown("Implementación manual sin el uso de librerías de caja negra.")

# Crear las 3 pestañas para la web
tab1, tab2, tab3 = st.tabs(["📈 1. Regresión Lineal", "📍 2. K-Nearest Neighbors", "📊 3. Naive Bayes"])

# ================= PESTAÑA 1: REGRESIÓN LINEAL =================
with tab1:
    st.header("Modelo de Regresión Lineal Simple")
    
    metodo_reg = st.radio("Selecciona el método para ingresar los datos de entrenamiento:", 
                          ["✏️ Ingreso Manual (Teclado)", "📂 Subir archivo CSV"], key="radio_reg")
    
    col_izq, col_der = st.columns([1, 2])
    df_reg = None

    with col_izq:
        if metodo_reg == "✏️ Ingreso Manual (Teclado)":
            st.write("Escribe los datos en la tabla (puedes agregar filas al final):")
            df_default_reg = pd.DataFrame({'X': [1.0, 2.0, 3.0, 4.0, 5.0], 'Y': [2.0, 4.0, 5.0, 4.0, 5.0]})
            df_reg = st.data_editor(df_default_reg, num_rows="dynamic", key="editor_reg")
        else:
            archivo_reg = st.file_uploader("Sube tu archivo CSV para Regresión", type=["csv"], key="file_reg")
            if archivo_reg is not None:
                df_reg = pd.read_csv(archivo_reg)
                st.write("Vista previa de los datos cargados:")
                st.dataframe(df_reg, use_container_width=True) # Muestra el cuadro de datos del CSV

        st.divider()
        st.subheader("Predicción Manual")
        nuevo_x = st.number_input("Ingresa un valor de X para predecir Y:", value=0.0)

    # Solo calcula si hay datos válidos (ya sea de teclado o CSV)
    if df_reg is not None and not df_reg.empty and len(df_reg.columns) >= 2:
        x = df_reg.iloc[:, 0].values
        y = df_reg.iloc[:, 1].values

        m, b = modelos.calcular_parametros(x, y)
        y_pred = modelos.predecir(x, m, b)
        mse = modelos.calcular_mse(y, y_pred)

        with col_izq:
            st.success(f"**Ecuación:** y = {m:.4f}x + {b:.4f}")
            st.info(f"**Error (MSE):** {mse:.4f}")
            pred_y = modelos.predecir(nuevo_x, m, b)
            st.warning(f"Si X = {nuevo_x}, la predicción es **Y = {pred_y:.4f}**")

        with col_der:
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='blue', label='Datos reales')
            ax.plot(x, y_pred, color='red', label='Línea de tendencia')
            ax.set_xlabel(df_reg.columns[0])
            ax.set_ylabel(df_reg.columns[1])
            ax.set_title("Regresión Lineal Simple")
            ax.legend()
            st.pyplot(fig)

# ================= PESTAÑA 2: K-NN =================
with tab2:
    st.header("Clasificador K-Nearest Neighbors (K-NN)")

    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.number_input("Valor de K (número de vecinos):", min_value=1, value=3, step=1)
    with col2:
        nx = st.number_input("Coordenada X del nuevo punto:", value=5.0)
    with col3:
        ny = st.number_input("Coordenada Y del nuevo punto:", value=4.0)

    st.divider()
    metodo_knn = st.radio("Selecciona el método para ingresar los datos de entrenamiento:", 
                          ["✏️ Ingreso Manual (Teclado)", "📂 Subir archivo CSV"], key="radio_knn")
    
    df_knn = None
    if metodo_knn == "✏️ Ingreso Manual (Teclado)":
        st.write("Escribe las coordenadas y la clase (0 o 1) en la tabla:")
        df_default_knn = pd.DataFrame({'X': [1.0, 2.0, 6.0, 7.0], 'Y': [2.0, 3.0, 7.0, 8.0], 'Clase': [0, 0, 1, 1]})
        df_knn = st.data_editor(df_default_knn, num_rows="dynamic", key="editor_knn")
    else:
        archivo_knn = st.file_uploader("Sube tu archivo CSV para K-NN", type=["csv"], key="file_knn")
        if archivo_knn is not None:
            df_knn = pd.read_csv(archivo_knn)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df_knn, use_container_width=True) # Muestra el cuadro de datos del CSV

    if df_knn is not None and not df_knn.empty and len(df_knn.columns) >= 3:
        x = df_knn.iloc[:, 0].values
        y = df_knn.iloc[:, 1].values
        clases = df_knn.iloc[:, 2].values
        datos_entrenamiento = np.column_stack((x, y))
        nuevo_punto = [nx, ny]

        resultado, vecinos = modelos.clasificar_knn(datos_entrenamiento, clases, nuevo_punto, k)

        st.success(f"¡El punto ({nx}, {ny}) fue clasificado en la **CLASE {resultado}** mediante votación!")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(x, y, c=clases, cmap='coolwarm', edgecolors='k', s=100, label='Datos Entrenamiento')
        ax2.scatter(nx, ny, color='yellow', marker='*', s=300, edgecolors='k', label='Nuevo Punto')

        distancias_plot = []
        for i in range(len(datos_entrenamiento)):
            d = np.sqrt(np.sum((np.array(nuevo_punto) - datos_entrenamiento[i])**2))
            distancias_plot.append((d, x[i], y[i]))
        distancias_plot.sort(key=lambda item: item[0])
        vecinos_plot = distancias_plot[:k]

        for idx, (_, vx, vy) in enumerate(vecinos_plot):
            etiqueta = 'Vecino Cercano' if idx == 0 else ""
            ax2.scatter(vx, vy, facecolors='none', edgecolors='green', s=250, linewidth=3, label=etiqueta)

        ax2.set_xlabel(df_knn.columns[0])
        ax2.set_ylabel(df_knn.columns[1])
        ax2.set_title(f"Clasificación K-NN (K={k})")
        
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())
        st.pyplot(fig2)

# ================= PESTAÑA 3: NAIVE BAYES =================
with tab3:
    st.header("Clasificador Naive Bayes")
    
    metodo_nb = st.radio("Selecciona el método para ingresar los datos de entrenamiento:", 
                          ["✏️ Ingreso Manual (Teclado)", "📂 Subir archivo CSV"], key="radio_nb")
    
    df_nb = None
    if metodo_nb == "✏️ Ingreso Manual (Teclado)":
        st.write("Escribe las categorías en la tabla (La última columna es la clase a predecir):")
        df_default_nb = pd.DataFrame({'Clima': ['Soleado', 'Nublado', 'Lluvia', 'Soleado'], 
                                      'Temperatura': ['Calor', 'Calor', 'Frio', 'Frio'], 
                                      'Jugar': ['No', 'Si', 'Si', 'Si']})
        df_nb = st.data_editor(df_default_nb, num_rows="dynamic", key="editor_nb")
    else:
        archivo_nb = st.file_uploader("Sube tu archivo CSV para Naive Bayes", type=["csv"], key="file_nb")
        if archivo_nb is not None:
            df_nb = pd.read_csv(archivo_nb)
            st.write("Vista previa de los datos cargados:")
            # Esto garantiza que el cuadro del CSV se muestre siempre
            st.dataframe(df_nb, use_container_width=True)

    if df_nb is not None and not df_nb.empty:
        df_nb = df_nb.astype(str)
        X_train = df_nb.iloc[:, :-1].values
        y_train = df_nb.iloc[:, -1].values
        columnas_X = df_nb.columns[:-1]

        clases, prioris, tablas = modelos.entrenar_naive_bayes(X_train, y_train)
        predicciones = [modelos.clasificar_naive_bayes(clases, prioris, tablas, inst) for inst in X_train]
        accuracy = np.sum(predicciones == y_train) / len(y_train) * 100

        st.success("¡Modelo entrenado con los datos de la tabla superior! ✅")
        st.metric(label="Precisión del modelo entrenado (Accuracy)", value=f"{accuracy:.2f}%")

        st.divider()
        st.subheader("Predicción Manual de un Nuevo Caso")
        st.write("Ingresa los datos para realizar una nueva predicción basada en el modelo:")

        nueva_instancia = []
        cols_input = st.columns(len(columnas_X))
        
        for i, col_name in enumerate(columnas_X):
            with cols_input[i]:
                valor = st.text_input(f"{col_name}:", key=f"input_nb_{i}")
                nueva_instancia.append(valor)

        if st.button("Predecir Clase", key="btn_nb"):
            if all(nueva_instancia):
                pred_nb = modelos.clasificar_naive_bayes(clases, prioris, tablas, nueva_instancia)
                st.warning(f"La nueva instancia fue clasificada como: **{pred_nb}**")
            else:
                st.error("Por favor, llena todos los campos para hacer la predicción.")
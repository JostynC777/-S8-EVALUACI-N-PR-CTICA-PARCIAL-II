import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import modelos

# Variable global para guardar el modelo de regresión
modelo_reg = {'m': None, 'b': None}

# --- FUNCIÓN PESTAÑA 1: REGRESIÓN LINEAL ---
def cargar_datos_y_calcular_regresion():
    ruta_archivo = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")))
    if not ruta_archivo: return
        
    try:
        df = pd.read_csv(ruta_archivo)
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        
        m, b = modelos.calcular_parametros(x, y)
        
        # Guardamos m y b para poder predecir después
        modelo_reg['m'] = m
        modelo_reg['b'] = b
        
        y_pred = modelos.predecir(x, m, b)
        mse = modelos.calcular_mse(y, y_pred)
        
        etiqueta_resultados_reg.config(text=f"Ecuación: y = {m:.4f}x + {b:.4f}\nError (MSE): {mse:.4f}")
        
        plt.figure("Regresión Lineal")
        plt.clf() # CORRECCIÓN: Limpia el gráfico anterior para que no se superpongan
        plt.scatter(x, y, color='blue', label='Datos reales')
        plt.plot(x, y_pred, color='red', label='Línea de tendencia')
        plt.xlabel("Variable Independiente (X)")
        plt.ylabel("Variable Dependiente (Y)")
        plt.title("Regresión Lineal Simple")
        plt.legend()
        plt.show(block=False) # CORRECCIÓN: Permite interactuar con la ventana principal sin cerrarla
        
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un problema:\n{e}")

# Nueva función para predecir manualmente
def predecir_manual_regresion():
    if modelo_reg['m'] is None:
        messagebox.showwarning("Aviso", "Primero carga un CSV y calcula el modelo.")
        return
    try:
        nuevo_x = float(entrada_x_reg.get())
        pred_y = modelos.predecir(nuevo_x, modelo_reg['m'], modelo_reg['b'])
        etiqueta_prediccion_reg.config(text=f"Si X = {nuevo_x}, la predicción es Y = {pred_y:.4f}")
    except ValueError:
        messagebox.showerror("Error", "Ingresa un número válido para X.")

# --- FUNCIÓN PESTAÑA 2: K-NN ---
def cargar_y_clasificar_knn():
    ruta_archivo = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")))
    if not ruta_archivo: return
    
    try:
        k = int(entrada_k.get())
        nx = float(entrada_x.get())
        ny = float(entrada_y.get())
        nuevo_punto = [nx, ny]

        df = pd.read_csv(ruta_archivo)
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        clases = df.iloc[:, 2].values
        datos_entrenamiento = np.column_stack((x, y))

        resultado, vecinos = modelos.clasificar_knn(datos_entrenamiento, clases, nuevo_punto, k)
        etiqueta_resultados_knn.config(text=f"¡El punto ({nx}, {ny}) fue clasificado en la CLASE {resultado}!")

        plt.figure("Clasificación K-NN")
        plt.clf() # CORRECCIÓN: Limpia el gráfico anterior
        plt.scatter(x, y, c=clases, cmap='coolwarm', edgecolors='k', s=100, label='Datos de Entrenamiento')
        plt.scatter(nx, ny, color='yellow', marker='*', s=300, edgecolors='k', label='Nuevo Punto')
        
        distancias_plot = []
        for i in range(len(datos_entrenamiento)):
            d = np.sqrt(np.sum((np.array(nuevo_punto) - datos_entrenamiento[i])**2))
            distancias_plot.append((d, x[i], y[i]))
        distancias_plot.sort(key=lambda item: item[0])
        vecinos_plot = distancias_plot[:k]
        
        for idx, (_, vx, vy) in enumerate(vecinos_plot):
            etiqueta = 'Vecino Cercano' if idx == 0 else ""
            plt.scatter(vx, vy, facecolors='none', edgecolors='green', s=250, linewidth=3, label=etiqueta)
        
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.title(f"Clasificación K-NN (K={k})")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show(block=False) # CORRECCIÓN: Permite interactuar con la ventana principal sin cerrarla

    except ValueError:
        messagebox.showerror("Error", "Asegúrate de ingresar números válidos para K, X e Y.")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un problema:\n{e}")

# --- FUNCIÓN PESTAÑA 3: NAIVE BAYES ---
def ejecutar_naive_bayes():
    ruta_archivo = filedialog.askopenfilename(title="Seleccionar CSV para Naive Bayes", filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")))
    if not ruta_archivo: return
    
    try:
        df = pd.read_csv(ruta_archivo)
        df = df.astype(str)
        
        X = df.iloc[:, :-1].values 
        y = df.iloc[:, -1].values  
        
        clases, prioris, tablas = modelos.entrenar_naive_bayes(X, y)
        
        predicciones = [modelos.clasificar_naive_bayes(clases, prioris, tablas, inst) for inst in X]
        accuracy = np.sum(predicciones == y) / len(y) * 100
        
        etiqueta_resultados_nb.config(text=f"Modelo Entrenado Exitosamente.\nPrecisión (Accuracy): {accuracy:.2f}%")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error en Naive Bayes:\n{e}")

# --- DISEÑO DE LA VENTANA PRINCIPAL ---
ventana = tk.Tk()
ventana.title("Proyecto IA - Algoritmos")
ventana.geometry("550x550")

notebook = ttk.Notebook(ventana)
notebook.pack(pady=10, expand=True, fill='both')

# --- PESTAÑA 1 ---
frame_reg = ttk.Frame(notebook)
notebook.add(frame_reg, text="1. Regresión Lineal")
tk.Label(frame_reg, text="Modelo de Regresión Lineal", font=("Arial", 14, "bold")).pack(pady=15)
tk.Button(frame_reg, text="Cargar CSV y Calcular", command=cargar_datos_y_calcular_regresion, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)
etiqueta_resultados_reg = tk.Label(frame_reg, text="Carga un archivo CSV para ver los resultados.", font=("Arial", 11))
etiqueta_resultados_reg.pack(pady=10)

frame_pred_reg = tk.Frame(frame_reg)
frame_pred_reg.pack(pady=15)
tk.Label(frame_pred_reg, text="Ingresa un nuevo valor de X:").grid(row=0, column=0, padx=5)
entrada_x_reg = tk.Entry(frame_pred_reg, width=10)
entrada_x_reg.grid(row=0, column=1, padx=5)
tk.Button(frame_pred_reg, text="Predecir Y", command=predecir_manual_regresion, bg="#8BC34A").grid(row=0, column=2, padx=5)

etiqueta_prediccion_reg = tk.Label(frame_reg, text="", font=("Arial", 12, "bold"), fg="blue")
etiqueta_prediccion_reg.pack(pady=5)

# --- PESTAÑA 2 ---
frame_knn = ttk.Frame(notebook)
notebook.add(frame_knn, text="2. K-Nearest Neighbors")
tk.Label(frame_knn, text="Clasificador K-NN", font=("Arial", 14, "bold")).pack(pady=10)
frame_inputs = tk.Frame(frame_knn)
frame_inputs.pack(pady=10)
tk.Label(frame_inputs, text="Valor de K (vecinos):").grid(row=0, column=0, padx=5, pady=5)
entrada_k = tk.Entry(frame_inputs, width=10)
entrada_k.grid(row=0, column=1, padx=5, pady=5)
entrada_k.insert(0, "3")
tk.Label(frame_inputs, text="Coordenada X a clasificar:").grid(row=1, column=0, padx=5, pady=5)
entrada_x = tk.Entry(frame_inputs, width=10)
entrada_x.grid(row=1, column=1, padx=5, pady=5)
tk.Label(frame_inputs, text="Coordenada Y a clasificar:").grid(row=2, column=0, padx=5, pady=5)
entrada_y = tk.Entry(frame_inputs, width=10)
entrada_y.grid(row=2, column=1, padx=5, pady=5)
tk.Button(frame_knn, text="Cargar CSV y Clasificar", command=cargar_y_clasificar_knn, bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=10)
etiqueta_resultados_knn = tk.Label(frame_knn, text="Ingresa las coordenadas y carga un CSV.", font=("Arial", 11))
etiqueta_resultados_knn.pack(pady=10)

# --- PESTAÑA 3 ---
frame_nb = ttk.Frame(notebook)
notebook.add(frame_nb, text="3. Naive Bayes")
tk.Label(frame_nb, text="Clasificador Naive Bayes", font=("Arial", 14, "bold")).pack(pady=15)
tk.Button(frame_nb, text="Cargar CSV y Entrenar", command=ejecutar_naive_bayes, bg="#FF9800", fg="white", font=("Arial", 12)).pack(pady=10)
etiqueta_resultados_nb = tk.Label(frame_nb, text="El sistema mostrará la precisión del modelo.", font=("Arial", 11))
etiqueta_resultados_nb.pack(pady=20)

ventana.mainloop()
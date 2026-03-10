import numpy as np

def calcular_parametros(x, y):
    """
    Calcula la pendiente (m) y el intercepto (b) usando Mínimos Cuadrados Ordinarios.
    """
    x = np.array(x)
    y = np.array(y)
    
    media_x = np.mean(x)
    media_y = np.mean(y)
    
    numerador = np.sum((x - media_x) * (y - media_y))
    denominador = np.sum((x - media_x)**2)
    m = numerador / denominador
    
    b = media_y - (m * media_x)
    
    return m, b

def calcular_mse(y_real, y_pred):
    """
    Calcula el Error Cuadrático Medio (MSE).
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_real - y_pred)**2)
    return mse

def predecir(x, m, b):
    """
    Realiza una predicción dado un valor de x, la pendiente y el intercepto.
    """
    x = np.array(x)
    return (m * x) + b

# --- FUNCIONES PARA K-NN ---

def distancia_euclidiana(punto1, punto2):
    """
    Calcula la distancia Euclidiana entre dos puntos.
    """
    punto1 = np.array(punto1)
    punto2 = np.array(punto2)
    return np.sqrt(np.sum((punto1 - punto2)**2))

def clasificar_knn(datos_entrenamiento, etiquetas, nuevo_punto, k):
    """
    Implementa el algoritmo K-NN desde cero.
    """
    distancias = []
    
    # 1. Calcular distancia del nuevo punto a todos los demás 
    for i in range(len(datos_entrenamiento)):
        d = distancia_euclidiana(nuevo_punto, datos_entrenamiento[i])
        distancias.append((d, etiquetas[i]))
    
    # 2. Ordenar por distancia (de menor a mayor)
    distancias.sort(key=lambda x: x[0])
    
    # 3. Seleccionar los K vecinos más cercanos
    vecinos_cercanos = distancias[:k]
    
    # 4. Votación de etiquetas
    votos = {}
    for _, etiqueta in vecinos_cercanos:
        votos[etiqueta] = votos.get(etiqueta, 0) + 1
    
    # Retornar la etiqueta con más votos
    resultado = max(votos, key=votos.get)
    return resultado, vecinos_cercanos

def entrenar_naive_bayes(X, y):
    """
    Crea las tablas de frecuencia y probabilidad para variables discretas.
    """
    X = np.array(X)
    y = np.array(y)
    clases = np.unique(y)
    prioris = {c: np.sum(y == c) / len(y) for c in clases}
    
    # Tabla de frecuencias: {clase: {columna: {valor: probabilidad}}}
    tablas = {}
    for c in clases:
        X_c = X[y == c]
        tablas[c] = {}
        for col in range(X.shape[1]):
            valores, conteos = np.unique(X_c[:, col], return_counts=True)
            total = len(X_c)
            # Aplicamos suavizado de Laplace (opcional pero recomendado)
            tablas[c][col] = {v: conteos[i] / total for i, v in enumerate(valores)}
            
    return clases, prioris, tablas

def clasificar_naive_bayes(clases, prioris, tablas, instancia):
    """
    Calcula la Probabilidad Posterior para clasificar una nueva instancia.
    """
    posteriores = {}
    for c in clases:
        # P(Clase)
        prob_posterior = prioris[c]
        # P(Clase) * P(Característica | Clase)
        for col, valor in enumerate(instancia):
            prob_condicional = tablas[c][col].get(valor, 0.0001) # 0.0001 si el valor no existe
            prob_posterior *= prob_condicional
        posteriores[c] = prob_posterior
        
    return max(posteriores, key=posteriores.get)


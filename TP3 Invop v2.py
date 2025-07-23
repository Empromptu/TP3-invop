import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def W(x, puntos, pesos):
    dist = np.linalg.norm(puntos - x, axis=1)
    dist = np.where(dist < 1e-10, 1e-10, dist)
    return np.sum(pesos * dist)


##########################
##### Hooke y Jeeves #####
##########################

# funcion para correr dentro de Hooke y Jeeves
def explorar(f, base, delta):
    # ej depunto en R3: [1,2,3] 
    punto = base.copy()
    n = len(punto) 
    for i in range(n): 
        # evaluo cuanto vale el punto en f 
        f_actual = f(punto) 
        # hago un paso en la dimension i del punto [1 + delta, 2, 3] 
        punto[i] += delta   
        # evaluo el punto con el paso en f
        f_nuevo = f(punto)   
        # si el punto nuevo es mayor que el orignial, no me sirve.
        if f_nuevo >= f_actual:
            # pruebo con hacer el paso para el otro lado
            # [1 - delta, 2, 3]
            punto[i] -= 2 * delta
            f_nuevo = f(punto)
            # si tampoco es menor que el orignial lo descarto.
            if f_nuevo >= f_actual:
                punto[i] += delta  # vuelvo al punto [1,2,3]
    return punto

def hookejeeves(f,puntos, delta=0.5, alpha=2.0, epsilon=1e-5, max_iter=1000):
    # elijo el punto del medio por ejemplo
    x0 = puntos[len(puntos)//2].copy()  
    xe = explorar(f,x0,delta)

    i = 0
    while delta > epsilon and i < max_iter:
        
        if f(xe) < f(x0):
            # pattern move : Es un salto más largo en la dirección 
            # de la mejora encontrada.
            # porque si encontraste una dirección que mejora la función, tiene 
            # sentido suponer que seguir avanzando en esa dirección podría seguir mejorando.
            xp = [xe[i] + alpha * (xe[i] - x0[i]) for i in range(len(x0))]
            x0 = xe.copy() # cambio el x0 por el nuevo punto (f(xe)<f(x0))
            xe = explorar(f,xp,delta) # busco si el punto lejano es menor
        
        else:
            delta = delta/2  # reduzco el tamanio del salto
            xe = explorar(f,x0,delta) # busco en un punto mas cercano al x0

        i += 1

    return x0, f(x0)


###########################
## Gradiente descendente ##
###########################

def gradiente_armijo(puntos, pesos, alpha_init=1.0, beta=0.5, sigma=1e-4, tol=1e-6, max_iter=1000):
    """
    Gradiente descendente con búsqueda de paso tipo Armijo,
    implementado directamente para la función de Fermat-Weber ponderada:
    
        W(x) = sum_i w_i * ||x - a_i||

    Parámetros:
        puntos: array (n, d) de los puntos a_i
        pesos: array (n,) de los pesos w_i
        x0: punto inicial (array de dimensión d)
        alpha_init, beta, sigma: parámetros de búsqueda de Armijo
        tol: tolerancia en norma del gradiente
        max_iter: máximo número de iteraciones

    Retorna:
        x: solución final
        iteraciones: número de iteraciones realizadas
        historial: lista de valores de la función objetivo
    """
    x = puntos[0]
    historial = []

    for k in range(1, max_iter + 1):
        # Calcular W(x)
        diff = x - puntos
        dist = np.linalg.norm(diff, axis=1)
        dist = np.where(dist < 1e-10, 1e-10, dist)  # evitar división por 0
        fx = np.sum(pesos * dist)
        historial.append(fx)

        # Calcular gradiente de W(x) directamente
        grad = np.sum((pesos[:, None] * diff) / dist[:, None], axis=0)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            return x, historial

        d = -grad  # dirección de descenso
        alpha = alpha_init

        # Búsqueda de paso tipo Armijo
        while True:
            x_new = x + alpha * d
            diff_new = x_new - puntos
            dist_new = np.linalg.norm(diff_new, axis=1)
            dist_new = np.where(dist_new < 1e-10, 1e-10, dist_new)
            fx_new = np.sum(pesos * dist_new)

            if fx_new <= fx + sigma * alpha * np.dot(grad, d):
                break
            alpha *= beta

        x = x + alpha * d

    return x,  historial

####################
#### WEISZFELD #####
####################

def weiszfeld(puntos, pesos, tol=1e-6, max_iter=1000):
    """
    Algoritmo de Weiszfeld para minimizar la función de Fermat-Weber ponderada.
    
    Parámetros:
        puntos: ndarray de forma (n, d), donde cada fila es un punto en R^d
        pesos: ndarray de forma (n,), pesos positivos asociados a cada punto
        tol: tolerancia para la convergencia
        max_iter: máximo número de iteraciones permitidas

    Retorna:
        x: punto solución en R^d
        iteraciones: número de iteraciones realizadas
    """

    n, d = puntos.shape

    # Paso previo: ¿algún punto a_k es solución trivial?
    for k in range(n):
        diff = puntos[k] - puntos
        dist = np.linalg.norm(diff, axis=1)
        dist = np.where(dist < 1e-10, 1e-10, dist)
        grad = np.sum((pesos[:, None] * (puntos[k] - puntos)) / dist[:, None], axis=0)
        if np.linalg.norm(grad) <= pesos[k]:
            return puntos[k], 0

    # Inicialización: promedio ponderado
    x = np.sum((pesos[:, None] * puntos), axis=0) / np.sum(pesos)

    for it in range(1, max_iter + 1):
        diff = puntos - x
        dist = np.linalg.norm(diff, axis=1)
        dist = np.where(dist < 1e-10, 1e-10, dist)

        numerador = np.sum((pesos[:, None] * puntos) / dist[:, None], axis=0)
        denominador = np.sum(pesos / dist)
        x_new = numerador / denominador

        if np.linalg.norm(x_new - x) < tol:
            return x_new, it

        x = x_new

    return x, max_iter  # no convergió

###################
## EXPERIMENTOS ###
###################

import time

def generar_datos(distribucion='uniforme', n=30, d=2, semilla=None):
    if semilla is not None:
        np.random.seed(semilla)

    if distribucion == 'uniforme':
        puntos = np.random.uniform(0, 10, size=(n, d))
    elif distribucion == 'normal':
        puntos = np.random.normal(loc=5, scale=2, size=(n, d))
    elif distribucion == 'cluster':
        centro1 = np.zeros(d)
        centro2 = np.ones(d) * 10
        puntos = np.vstack([
            np.random.normal(centro1, 1.0, size=(n//2, d)),
            np.random.normal(centro2, 1.0, size=(n - n//2, d))
        ])
    else:
        raise ValueError("Distribución no reconocida")

    pesos = np.random.randint(1, 100, size=n)
    return puntos, pesos


def benchmark_grafico(n=310, d=2, repeticiones=5):
    distribuciones = ['uniforme', 'normal', 'cluster']
    tiempos_w, tiempos_hj, tiempos_gd = [], [], []

    for dist in distribuciones:
        tiempo_total_w = tiempo_total_hj = tiempo_total_gd = 0

        for _ in range(repeticiones):
            puntos, pesos = generar_datos(dist, n=n, d=d)

            # Weiszfeld
            t0 = time.perf_counter()
            _, _ = weiszfeld(puntos, pesos)
            tiempo_total_w += time.perf_counter() - t0

            # Hooke-Jeeves
            t0 = time.perf_counter()
            _, _ = hookejeeves(lambda x: W(x, puntos, pesos), puntos)
            tiempo_total_hj += time.perf_counter() - t0

            # Gradiente descendente
            t0 = time.perf_counter()
            _, _ = gradiente_armijo(puntos, pesos)
            tiempo_total_gd += time.perf_counter() - t0

        # Promedio
        tiempos_w.append(tiempo_total_w / repeticiones)
        tiempos_hj.append(tiempo_total_hj / repeticiones)
        tiempos_gd.append(tiempo_total_gd / repeticiones)

    # === Gráfico ===
    x = np.arange(len(distribuciones))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, tiempos_w, width, label='Weiszfeld')
    ax.bar(x, tiempos_hj, width, label='Hooke-Jeeves')
    ax.bar(x + width, tiempos_gd, width, label='Grad. Desc.')

    ax.set_ylabel('Tiempo promedio (segundos)')
    ax.set_title(f'Comparación de tiempos con n={n}, d={d}')
    ax.set_xticks(x)
    ax.set_xticklabels(distribuciones)
    ax.legend()
    plt.tight_layout()
    plt.show()

benchmark_grafico(n=500, d=10)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 10 puntos en R3
puntos = np.array([
    [ 1.2,  3.4,  5.6],
    [ 2.1,  0.4,  7.8],
    [ 4.5,  6.7,  1.2],
    [ 3.3,  2.2,  9.9],
    [ 0.0,  1.1,  2.2],
    [ 5.5,  4.4,  3.3],
    [ 6.6,  5.5,  4.4],
    [ 7.7,  8.8,  9.9],
    [ 9.0,  0.1,  1.2],
    [ 2.2,  3.3,  4.4]
])

pesos = np.array([35,22,23,14,66,84,51,12,9,65])

def W(x, puntos, pesos):
    dist = np.linalg.norm(puntos - x, axis=1)
    return np.sum(pesos * dist)

def W_grad(x, puntos, pesos):
    grad = np.zeros_like(x)
    dist = np.linalg.norm(puntos - x, axis=1)
    for i in range(len(puntos)):
        if dist[i] != 0:
            grad += pesos[i] * (x - puntos[i]) / dist[i]
    return grad

def weiszfeld(puntos, pesos, alpha=1e-6, max_iter=1000):
    m, d = puntos.shape

    # Paso previo: algun punto a_k es optimo?
    for k in range(m):
        gk = W_grad(puntos[k], puntos, pesos)
        if np.linalg.norm(gk) <= pesos[k]:
            return puntos[k], 0  # Es un vertice optimo

    # Criterio de inicio: elijo un punto de inicio que es un promedio ponderado.
    x = np.sum((pesos[:, None] * puntos), axis=0) / np.sum(pesos)
    k = 0

    while k < max_iter:

        grad = W_grad(x, puntos, pesos)
        w = W(x, puntos, pesos)
        Dk = w - np.dot(grad, x) + np.min([np.dot(grad, a) for a in puntos])  # medida de error

        # Criterio de parada:
        # Si la medida de error (Dk) es menor al alpha, entonces estoy bastante cerca del mejor punto
        if Dk / np.sum(pesos) <= alpha:
            return x, k

        # Aplico la transformacion de Weiszfeld
        dist = np.linalg.norm(puntos - x, axis=1)
        numerador = np.sum((pesos[:, None] * puntos) / dist[:, None], axis=0)
        denominador = np.sum(pesos / dist)
        x = numerador / denominador

        k += 1

    return x, k  # retorno aunque no haya alcanzado la tolerancia


solucion, iteraciones = weiszfeld(puntos, pesos)

# === GRAFICO ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos 
ax.scatter(puntos[:,0], puntos[:,1], puntos[:,2], c='blue', label='Puntos', s=pesos, alpha=0.6)

# Graficar el punto solución
ax.scatter(solucion[0], solucion[1], solucion[2], c='red', label='Solución (mínimo)', s=100)

# Líneas desde cada punto a la solución
for p in puntos:
    ax.plot([p[0], solucion[0]], [p[1], solucion[1]], [p[2], solucion[2]], 'gray', linestyle='dotted', alpha=0.5)

ax.set_title(f'Algoritmo de Weiszfeld en $\\mathbb{{R}}^3$\nIteraciones: {iteraciones}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()


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
            xe = explorar(x0,delta) # busco en un punto mas cercano al x0

        i += 1

    return x0, f(x0)
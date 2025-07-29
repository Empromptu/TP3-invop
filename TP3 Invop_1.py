import numpy as np
import csv
import time
import os

def benchmark_csv(n=300, d=2, distribuciones=['uniforme', 'normal', 'cluster'], archivo='resultados_benchmark.csv'):
    # Si el archivo existe, lo borro para no duplicar
    if os.path.exists(archivo):
        os.remove(archivo)

    # Crear encabezado
    with open(archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distribucion', 'n', 'd','algoritmo', 'tiempo', 'iteraciones','solucion'])

    # Funciones auxiliares necesarias
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

    def W(x, puntos, pesos):
        return np.sum(pesos * np.linalg.norm(puntos - x, axis=1))

    def W_grad(x, puntos, pesos):
        dif = x - puntos  # (n, d)
        dist = np.linalg.norm(dif, axis=1)
        dist = np.where(dist < 1e-10, 1e-10, dist)
        return np.sum(pesos[:, None] * dif / dist[:, None], axis=0)
        """
        grad = np.zeros_like(x)
        dist = np.linalg.norm(puntos - x, axis=1)
        dist = np.where(dist < 1e-10, 1e-10, dist)
        for i in range(len(puntos)):
            grad += pesos[i] * (x - puntos[i]) / dist[i]
        return grad
        """
    def gradiente_armijo(f, grad_f, x0, alpha_init=1.0, beta=0.5, sigma=1e-4, tol=1e-6, max_iter=1000):
        x = np.sum(pesos[:, None] * puntos, axis=0) / np.sum(pesos)
        prev_val = W(x, puntos, pesos)
        for k in range(1, max_iter + 1):
            grad = grad_f(x)
            if np.linalg.norm(grad) < tol:
                return x, k
            d = -grad
            alpha = alpha_init
            while f(x + alpha * d) > f(x) + sigma * alpha * np.dot(grad, d):
                alpha *= beta
            x = x + alpha * d

            new_val = W(x, puntos, pesos)
            if abs(prev_val - new_val) < tol:
                return x, k

            prev_val = new_val
        return x, max_iter

    def hookejeeves(f, puntos, delta=0.5, alpha=2.0, epsilon=1e-5, max_iter=1000):
        def explorar(f, base, delta):
            punto = base.copy()
            for i in range(len(punto)):
                f_actual = f(punto)
                punto[i] += delta
                f_nuevo = f(punto)
                if f_nuevo >= f_actual:
                    punto[i] -= 2 * delta
                    f_nuevo = f(punto)
                    if f_nuevo >= f_actual:
                        punto[i] += delta
            return punto

        x0 = puntos[len(puntos)//2].copy()
        xe = explorar(f, x0, delta)
        i = 0
        while delta > epsilon and i < max_iter:
            if f(xe) < f(x0):
                xp = [xe[i] + alpha * (xe[i] - x0[i]) for i in range(len(x0))]#xe + alpha * (xe - x0)
                x0 = xe.copy()
                xe = explorar(f, xp, delta)
            else:
                delta /= 2
                xe = explorar(f, x0, delta)
            i += 1
        return x0, i

    def weiszfeld(puntos, pesos, tol=1e-6, max_iter=1000):
        n, d = puntos.shape
        x = np.sum((pesos[:, None] * puntos), axis=0) / np.sum(pesos)
        for k in range(1, max_iter + 1):
            dist = np.linalg.norm(puntos - x, axis=1)
            if np.any(dist < 1e-10):
                j = np.argmin(dist)
                diffs = puntos - puntos[j]
                dist = np.linalg.norm(diffs, axis=1)
                mask = np.arange(n) != j
                dist = np.where(dist < 1e-10, 1e-10, dist)
                Rj = np.sum((pesos[mask, None] * (-diffs[mask]) / dist[mask, None]), axis=0)
                norm_Rj = np.linalg.norm(Rj)
                if norm_Rj < 1e-10:
                    return puntos[j], k
                dj = -Rj / norm_Rj
                denom = np.sum(pesos[mask] / dist[mask])
                tj = (norm_Rj - pesos[j]) / denom
                x = puntos[j] + dj * tj
                continue
            dist = np.where(dist < 1e-10, 1e-10, dist)
            num = np.sum((pesos[:, None] * puntos) / dist[:, None], axis=0)
            denom = np.sum(pesos / dist)
            x_new = num / denom
            if np.linalg.norm(x_new - x) < tol:
                return x_new, k
            x = x_new
        return x, max_iter

    # === Benchmarking ===
    for dist in distribuciones:
        puntos, pesos = generar_datos(dist, n=n, d=d, semilla=42)

        # --- Weiszfeld ---
        t0 = time.time()
        xw, it_w = weiszfeld(puntos, pesos)
        tw = time.time() - t0
        with open(archivo, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dist, n, d, 'weiszfeld', tw, it_w, list(xw)])

        # --- Hooke-Jeeves ---
        t0 = time.time()
        xhj, it_hj = hookejeeves(lambda x: W(x, puntos, pesos), puntos)
        thj = time.time() - t0
        with open(archivo, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dist, n, d, 'hooke-jeeves', thj, it_hj, list(xhj)])

        # --- Gradiente descendente ---
        t0 = time.time()
        xg, it_g = gradiente_armijo(lambda x: W(x, puntos, pesos),
                                    lambda x: W_grad(x, puntos, pesos),
                                    puntos[0])
        tg = time.time() - t0
        with open(archivo, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dist, n, d, 'gradiente', tg, it_g, list(xg)])

print("Procesando...")


benchmark_csv(n=100000, d=20, archivo='resultados_100000x20.csv')
print("listo archivo 10000x20")

"""
Dado una recurrencia lineal homogénea de coeficientes constantes de grado m:

    f(n) = a1*f(n-1) + a2*f(n-2) + ... + am*f(n-m)

con condiciones iniciales f(0)=C0, f(1)=C1, ..., f(m-1)=Cm-1,
este programa encuentra la expresión cerrada (no recurrente) para f(n)
y la evalúa en un punto n dado por el usuario.

Método:
    1. Se construye el polinomio característico: r^m - a1*r^(m-1) - ... - am = 0
    2. Se encuentran las raíces (con multiplicidades).
    3. Se plantea la solución general como combinación lineal de términos r^n
       (o n^k * r^n para raíces repetidas).
    4. Se resuelve el sistema lineal con las condiciones iniciales.
    5. Se imprime la expresión cerrada y se evalúa f(n).

Dependencias:
    - numpy  (raíces del polinomio y sistema lineal)
    - sympy  (manipulación simbólica para expresión cerrada)
"""

import numpy as np
import sympy as sp


# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────

def leer_entero(mensaje):
    """
    Lee un entero desde consola con manejo de excepción.
    Repite la solicitud hasta obtener un valor válido.
    """
    while True:
        try:
            return int(input(mensaje))
        except ValueError:
            print("ingrese un número entero válido.\n")


def leer_numero(mensaje):
    """
    Lee un número real (float) desde consola con manejo de excepción.
    Acepta también enteros. Repite si la entrada no es válida.
    """
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("ingrese un número válido (puede ser decimal).\n")


def obtener_raices_con_multiplicidad(coefs_poly, tol=1e-6):
    """
    Dado el arreglo de coeficientes del polinomio característico,
    calcula sus raíces numéricas y las agrupa por multiplicidad.

    Parámetros:
        coefs_poly : lista de coeficientes [1, -a1, -a2, ..., -am]
        tol        : tolerancia para considerar dos raíces iguales

    Retorna:
        lista de tuplas (raiz, multiplicidad)
    """
    raices_raw = np.roots(coefs_poly)

    agrupadas = []
    usada = [False] * len(raices_raw)

    for i, r in enumerate(raices_raw):
        if usada[i]:
            continue
        mult = 1
        usada[i] = True
        for j in range(i + 1, len(raices_raw)):
            if not usada[j] and abs(r - raices_raw[j]) < tol:
                mult += 1
                usada[j] = True
        agrupadas.append((r, mult))

    return agrupadas


def construir_sistema(raices_mult, m):
    """
    Construye la matriz del sistema lineal A·x = b que surge de imponer
    las m condiciones iniciales a la solución general.

    Cada raíz r con multiplicidad k aporta los términos:
        r^n, n*r^n, n^2*r^n, ..., n^(k-1)*r^n

    Evaluando en n = 0, 1, ..., m-1 se obtienen las filas de A.

    Parámetros:
        raices_mult : lista de (raiz, multiplicidad)
        m           : grado de la recurrencia

    Retorna:
        A : matriz numpy (m x m) compleja
    """
    A = np.zeros((m, m), dtype=complex)
    col = 0
    for (r, mult) in raices_mult:
        for k in range(mult):
            for fila in range(m):
                n = fila
                # término: n^k * r^n  (con 0^0 = 1 por convención)
                pot_n = (n ** k) if not (n == 0 and k == 0) else 1
                pot_r = (r ** n) if n != 0 else 1
                A[fila, col] = pot_n * pot_r
            col += 1
    return A


def imprimir_expresion_cerrada(raices_mult, coefs_solucion):
    """
    Imprime la expresión cerrada de f(n) de forma legible,
    mostrando cada término α_k * n^j * r^n.

    Parámetros:
        raices_mult    : lista de (raiz, multiplicidad)
        coefs_solucion : arreglo numpy con los coeficientes α resueltos
    """
    n = sp.Symbol('n')
    expr = sp.Integer(0)

    idx = 0
    terminos_str = []

    for (r, mult) in raices_mult:
        for k in range(mult):
            alpha = coefs_solucion[idx]
            # Redondear parte real e imaginaria pequeñas a cero
            re = round(alpha.real, 8)
            im = round(alpha.imag, 8)
            alpha_limpio = complex(re, im)

            # Construir representación del término
            if abs(alpha_limpio) < 1e-9:
                idx += 1
                continue

            coef_str = f"({alpha_limpio:.6g})"
            base_str = f"({r:.6g})^n"
            if k == 0:
                poly_str = ""
            elif k == 1:
                poly_str = "n * "
            else:
                poly_str = f"n^{k} * "

            terminos_str.append(f"  {coef_str} * {poly_str}{base_str}")
            idx += 1

    if terminos_str:
        print("\n  f(n) =")
        print("  +\n".join(terminos_str))
    else:
        print("\n  f(n) = 0")


def evaluar_fn(raices_mult, coefs_solucion, n_val):
    """
    Evalúa numéricamente f(n) usando la expresión cerrada.

    Parámetros:
        raices_mult    : lista de (raiz, multiplicidad)
        coefs_solucion : coeficientes α del sistema
        n_val          : valor entero de n a evaluar

    Retorna:
        Valor real de f(n_val)
    """
    resultado = 0 + 0j
    idx = 0
    for (r, mult) in raices_mult:
        for k in range(mult):
            alpha = coefs_solucion[idx]
            n_k = (n_val ** k) if not (n_val == 0 and k == 0) else 1
            r_n = (r ** n_val) if n_val != 0 else 1
            resultado += alpha * n_k * r_n
            idx += 1
    return round(resultado.real, 6)


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main():
    print("  SOLUCIÓN CERRADA DE RECURRENCIA LINEAL HOMOGÉNEA")
    print("  f(n) = a1·f(n-1) + a2·f(n-2) + ... + am·f(n-m)")

    # Leer m 
    while True:
        m = leer_entero("\nIngrese el grado m de la recurrencia: ")
        if m >= 1:
            break
        print("m debe ser un entero positivo >= 1.")

    # Leer coeficientes a1, a2, ..., am
    print(f"\n── Coeficientes de la recurrencia (a1 hasta a{m}) ──")
    ar = []
    for i in range(1, m + 1):
        a = leer_numero(f"  Ingrese a{i}: ")
        ar.append(a)

    # Leer condiciones iniciales C0, C1, ..., Cm-1
    print(f"\n── Condiciones iniciales (f(0) hasta f({m-1})) ──")
    cr = []
    for i in range(m):
        c = leer_numero(f"  Ingrese C{i} = f({i}): ")
        cr.append(c)

    # Leer n 
    while True:
        n_val = leer_entero("\nIngrese el valor de n para evaluar f(n): ")
        if n_val >= 0:
            break
        print("n debe ser un entero no negativo.")

    # Caso trivial: n es condición inicial
    if n_val < m:
        print(f"\n  f({n_val}) = {cr[n_val]}  (es una condición inicial directa)")
        return

    # Construir polinomio característico
    # r^m - a1*r^(m-1) - a2*r^(m-2) - ... - am = 0
    # coeficientes: [1, -a1, -a2, ..., -am]
    coefs_poly = [1] + [-a for a in ar]

    # Calcular raíces con multiplicidad
    try:
        raices_mult = obtener_raices_con_multiplicidad(coefs_poly)
    except Exception as e:
        print(f"\nError al calcular raíces: {e}")
        return

    print("\n── Raíces del polinomio característico ──")
    for (r, mult) in raices_mult:
        re = round(r.real, 6)
        im = round(r.imag, 6)
        if abs(im) < 1e-9:
            print(f"  r = {re}  (multiplicidad {mult})")
        else:
            print(f"  r = {re} + {im}i  (multiplicidad {mult})")

    # Plantear y resolver sistema lineal
    A = construir_sistema(raices_mult, m)
    b = np.array(cr, dtype=complex)

    try:
        # Verificar que A no es singular
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            raise np.linalg.LinAlgError("La matriz del sistema es singular (det ≈ 0).")
        coefs_solucion = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print(f"\nError al resolver el sistema lineal: {e}")
        print("  Verifique que las condiciones iniciales sean consistentes.")
        return

    # Mostrar expresión cerrada
    print("\n── Expresión cerrada ──")
    imprimir_expresion_cerrada(raices_mult, coefs_solucion)

    # Evaluar f(n)
    resultado = evaluar_fn(raices_mult, coefs_solucion, n_val)

    # Verificación cruzada iterativa
    f_iter = list(cr)
    for step in range(m, n_val + 1):
        fn_paso = sum(ar[i] * f_iter[-(i + 1)] for i in range(m))
        f_iter.append(fn_paso)
    resultado_iter = round(f_iter[n_val], 6)

    print(f"\n── Resultado ──")
    print(f"  f({n_val}) = {resultado}  (expresión cerrada)")
    print(f"  f({n_val}) = {resultado_iter}  (verificación iterativa)")

    if abs(resultado - resultado_iter) < 1e-3:
        print("Ambos métodos coinciden.")
    else:
        print("posible error numérico.")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Programa interrumpido por el usuario.")
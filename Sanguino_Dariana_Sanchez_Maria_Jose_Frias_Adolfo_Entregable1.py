"""
Dado una recurrencia lineal homogénea de coeficientes constantes de grado m:

    f(n) = a1*f(n-1) + a2*f(n-2) + ... + am*f(n-m)

con condiciones iniciales f(0)=C0, f(1)=C1, ..., f(m-1)=Cm-1,
este programa encuentra la expresión cerrada (no recurrente) para f(n)
y la evalúa en un punto n dado por el usuario.

Dependencias:
    - numpy  (raíces del polinomio y sistema lineal)
    - sympy  (manipulación simbólica para expresión cerrada)
"""

import numpy as np
import sympy as sp
import streamlit as st
import subprocess
import sys
import os
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
    st.title("Solución Cerrada de Recurrencia Lineal")
    st.markdown("### f(n) = a₁·f(n-1) + a₂·f(n-2) + ... + aₘ·f(n-m)")
    st.divider()
 
    # ── Leer m ──────────────────────────────────────────────
    m = st.number_input("Grado m de la recurrencia:", min_value=1, step=1, value=2)
 
    st.divider()
 
    # ── Leer coeficientes ────────────────────────────────────
    st.subheader("Coeficientes aᵢ")
    ar = []
    cols = st.columns(m)
    for i in range(m):
        with cols[i]:
            a = st.number_input(f"a{i+1}", value=1.0, key=f"a{i}")
            ar.append(a)
 
    st.divider()
 
    # ── Leer condiciones iniciales ───────────────────────────
    st.subheader("Condiciones iniciales Cᵢ")
    cr = []
    cols2 = st.columns(m)
    for i in range(m):
        with cols2[i]:
            c = st.number_input(f"C{i} = f({i})", value=float(i), key=f"c{i}")
            cr.append(c)
 
    st.divider()
 
    # ── Leer n ───────────────────────────────────────────────
    n_val = st.number_input("Valor de n a evaluar:", min_value=0, step=1, value=10)
 
    # ── Botón calcular ───────────────────────────────────────
    if st.button("Calcular", type="primary"):
 
        # Caso trivial
        if n_val < m:
            st.success(f"f({n_val}) = {cr[int(n_val)]}  (es una condición inicial directa)")
            return
 
        # Polinomio característico
        coefs_poly = [1] + [-a for a in ar]
 
        # Raíces
        try:
            raices_mult = obtener_raices_con_multiplicidad(coefs_poly)
        except Exception as e:
            st.error(f"Error al calcular raíces: {e}")
            return
 
        # Mostrar raíces
        st.subheader("Raíces del polinomio característico")
        for (r, mult) in raices_mult:
            re = round(r.real, 6)
            im = round(r.imag, 6)
            if abs(im) < 1e-9:
                st.write(f"r = {re}  (multiplicidad {mult})")
            else:
                st.write(f"r = {re} + {im}i  (multiplicidad {mult})")
 
        # Sistema lineal
        A = construir_sistema(raices_mult, m)
        b = np.array(cr, dtype=complex)
 
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                raise np.linalg.LinAlgError("La matriz del sistema es singular (det ≈ 0).")
            coefs_solucion = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            st.error(f"Error al resolver el sistema lineal: {e}")
            st.warning("Verifique que las condiciones iniciales sean consistentes.")
            return
 
        # Expresión cerrada
        st.subheader("Expresión cerrada")
        idx = 0
        terminos = []
        for (r, mult) in raices_mult:
            for k in range(mult):
                alpha = coefs_solucion[idx]
                re_a = round(alpha.real, 8)
                im_a = round(alpha.imag, 8)
                alpha_limpio = complex(re_a, im_a)
                if abs(alpha_limpio) > 1e-9:
                    coef_str = f"({alpha_limpio:.6g})"
                    base_str = f"({r:.6g})^n"
                    poly_str = "" if k == 0 else ("n · " if k == 1 else f"n^{k} · ")
                    terminos.append(f"{coef_str} · {poly_str}{base_str}")
                idx += 1
 
        if terminos:
            st.latex("f(n) = " + " + ".join(terminos))
        else:
            st.latex("f(n) = 0")
 
        # Resultados
        resultado = evaluar_fn(raices_mult, coefs_solucion, int(n_val))
 
        f_iter = list(cr)
        for step in range(m, int(n_val) + 1):
            fn_paso = sum(ar[i] * f_iter[-(i + 1)] for i in range(m))
            f_iter.append(fn_paso)
        resultado_iter = round(f_iter[int(n_val)], 6)
 
        st.divider()
        st.subheader("Resultado")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"f({int(n_val)}) — Expresión cerrada", value=resultado)
        with col2:
            st.metric(label=f"f({int(n_val)}) — Verificación iterativa", value=resultado_iter)
 
        if abs(resultado - resultado_iter) < 1e-3:
            st.success("Ambos métodos coinciden.")
        else:
            st.warning("Posible error numérico — los métodos difieren.")



# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        print("\n  use python -m streamlit run Sanguino_Dariana_Sanchez_Maria_Jose_Frias_Adolfo_Entregable1.py")
    except KeyboardInterrupt:
        print("\n\n  Programa interrumpido por el usuario.")

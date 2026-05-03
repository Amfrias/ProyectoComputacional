"""
Dado una recurrencia lineal homogénea de coeficientes constantes de grado m:

    f(n) = a1*f(n-1) + a2*f(n-2) + ... + am*f(n-m)

con condiciones iniciales f(0)=C0, f(1)=C1, ..., f(m-1)=Cm-1,
este programa encuentra la expresión cerrada (no recurrente) para f(n)
y la evalúa en un punto n dado por el usuario.

Dependencias:
    - numpy  (raíces del polinomio y sistema lineal)
    - sympy  (manipulación simbólica para expresión cerrada)
    - streamlit (interfaz gráfica en el navegador)
"""

import numpy as np       # para calcular raíces del polinomio y resolver el sistema lineal
import sympy as sp       # para manejar símbolos matemáticos en la expresión cerrada
import streamlit as st   # para construir la interfaz gráfica web
import sys
from streamlit.web import cli
from streamlit import runtime

# AUX FUNC:

def validar_coeficientes(ar, m):
    # comprobamos que la cantidad de coeficientes coincide con el grado m
    if len(ar) != m:
        raise ValueError(f"Se esperaban {m} coeficientes pero se recibieron {len(ar)}.")
    # revisamos uno por uno que no sean nan ni infinito
    for i, a in enumerate(ar):
        if not np.isfinite(a):
            raise ValueError(f"El coeficiente a{i+1} no es un número finito (valor={a}).")


def validar_condiciones_iniciales(cr, m):
    # misma cantidad que el grado de la recurrencia
    if len(cr) != m:
        raise ValueError(f"Se esperaban {m} condiciones iniciales pero se recibieron {len(cr)}.")
    # revisamos que ninguna sea nan ni infinito
    for i, c in enumerate(cr):
        if not np.isfinite(c):
            raise ValueError(f"La condición inicial C{i} no es un número finito (valor={c}).")


def obtener_raices_con_multiplicidad(coefs_poly, tol=1e-6):
    # numpy.roots nos devuelve las raíces del polinomio (puede haber complejas)
    raices_raw = np.roots(coefs_poly)

    agrupadas = []          # aquí vamos a guardar las raíces sin repetir
    usada = [False] * len(raices_raw)   # marcamos cuáles ya procesamos

    # recorremos todas las raíces para agrupar las que sean prácticamente iguales
    for i, r in enumerate(raices_raw):
        if usada[i]:
            # esta raíz ya fue agrupada antes, la saltamos
            continue
        mult = 1        # empezamos con multiplicidad 1
        usada[i] = True
        # buscamos otras raíces parecidas para sumarles a la multiplicidad
        for j in range(i + 1, len(raices_raw)):
            if not usada[j] and abs(r - raices_raw[j]) < tol:
                # son suficientemente cercanas, las contamos como la misma
                mult += 1
                usada[j] = True
        agrupadas.append((r, mult))   # guardamos (raíz, cuántas veces aparece)

    return agrupadas


def obtener_raices_exactas(coefs_poly):
    x = sp.Symbol('x')
    # convertimos cada coeficiente float a fracción exacta para que sympy trabaje bien
    coefs_sp = [sp.Rational(c).limit_denominator(10**9) if isinstance(c, float)
                else sp.Integer(c) for c in coefs_poly]

    # armamos el polinomio simbólico: c0*x^m + c1*x^(m-1) + ... + cm
    grado = len(coefs_sp) - 1
    poly_expr = sum(coefs_sp[i] * x**(grado - i) for i in range(len(coefs_sp)))

    try:
        # sp.roots devuelve {raiz: multiplicidad} con formas exactas si puede factorizar
        raices_exactas = sp.roots(poly_expr, x)
        if not raices_exactas:
            # sympy devuelve dict vacío si el polinomio no tiene raíces cerradas conocidas
            return None
        return raices_exactas
    except Exception:
        # si sympy falla por cualquier motivo, volvemos al modo numérico
        return None


def construir_sistema(raices_mult, m):
    # creamos la matriz de ceros compleja de tamaño m x m
    A = np.zeros((m, m), dtype=complex)
    col = 0   # columna actual que vamos llenando

    for (r, mult) in raices_mult:
        # por cada raíz con su multiplicidad generamos las columnas correspondientes
        for k in range(mult):
            # cada fila corresponde a evaluar en n = 0, 1, ..., m-1
            for fila in range(m):
                n = fila   # el valor de n para esta fila

                # calculamos n^k con el caso especial 0^0 = 1 (por definición matemática)
                pot_n = (n ** k) if not (n == 0 and k == 0) else 1

                # calculamos r^n con el caso especial r^0 = 1
                pot_r = (r ** n) if n != 0 else 1

                # el término que va en esta celda es n^k * r^n
                A[fila, col] = pot_n * pot_r
            col += 1   # avanzamos a la siguiente columna

    return A


def resolver_coeficientes_exactos(raices_exactas_dict, cr):
    m = len(cr)   # cantidad de condiciones iniciales = grado de la recurrencia

    # creamos un símbolo alpha_i por cada incógnita del sistema
    alphas = [sp.Symbol(f'alpha_{i}') for i in range(m)]

    # expandimos raíces respetando multiplicidad: (raíz, potencia_de_n)
    terminos_por_columna = []
    for r_sp, mult in raices_exactas_dict.items():
        for j in range(mult):
            terminos_por_columna.append((r_sp, j))

    # armamos las m ecuaciones evaluando la solución general en n = 0, 1, ..., m-1
    ecuaciones = []
    for fila in range(m):
        nval = fila
        expr = sum(
            alphas[col] * (sp.Integer(1) if (nval == 0 and j == 0) else sp.Integer(nval)**j)
                        * (sp.Integer(1) if nval == 0 else r_sp**nval)
            for col, (r_sp, j) in enumerate(terminos_por_columna)
        )
        # el lado derecho es la condición inicial C_fila convertida a racional
        ecuaciones.append(sp.Eq(expr, sp.Rational(cr[fila]).limit_denominator(10**9)))

    try:
        solucion = sp.solve(ecuaciones, alphas)
        if not solucion:
            return None
        # devolvemos los coeficientes en el mismo orden que la lista alphas
        return [solucion[a] for a in alphas]
    except Exception:
        return None


def evaluar_fn(raices_mult, coefs_solucion, n_val):
    resultado = 0 + 0j   # acumulador, usamos complejo por si las raíces son complejas
    idx = 0              # índice para recorrer coefs_solucion

    for (r, mult) in raices_mult:
        for k in range(mult):
            alpha = coefs_solucion[idx]   # coeficiente que encontramos al resolver el sistema

            # de nuevo tratamos el caso especial n=0, k=0 para no calcular 0^0 ambiguamente
            n_k = (n_val ** k) if not (n_val == 0 and k == 0) else 1
            r_n = (r ** n_val) if n_val != 0 else 1

            # sumamos el aporte de este término: alpha * n^k * r^n
            resultado += alpha * n_k * r_n
            idx += 1

    # la parte imaginaria debería ser cero (o muy chica) si los datos son reales
    return round(resultado.real, 6)


def limpiar_alpha(alpha):
    re = round(alpha.real, 8)
    im = round(alpha.imag, 8)
    # si la parte imaginaria es prácticamente cero, nos quedamos solo con el real
    if abs(im) < 1e-9:
        return re   # devolvemos float, ya no cargamos la parte imaginaria
    return complex(re, im)


def formatear_termino_latex(alpha_val, r_val, k, es_primero):
    # convertimos a LaTeX usando sympy para que salga bonito (sqrt, fracciones, etc.)
    if isinstance(alpha_val, sp.Basic):
        alpha_latex = sp.latex(sp.simplify(alpha_val))
        alpha_float = float(alpha_val.evalf())
    else:
        # valor numérico: usamos nsimplify para intentar convertir 0.5 → 1/2, etc.
        alpha_float = float(alpha_val.real) if hasattr(alpha_val, 'real') else float(alpha_val)
        alpha_latex = sp.latex(sp.nsimplify(alpha_val, rational=False, tolerance=1e-8))

    if isinstance(r_val, sp.Basic):
        r_latex = sp.latex(sp.simplify(r_val))
    else:
        r_latex = sp.latex(sp.nsimplify(r_val, rational=False, tolerance=1e-8))

    # determinamos si el coeficiente es negativo para manejar el signo aparte
    negativo = alpha_float < 0

    # si es negativo, le quitamos el signo del latex del coeficiente porque lo ponemos afuera
    if negativo:
        # sp.latex de un número negativo ya incluye el '-', lo removemos para no duplicar
        alpha_latex = sp.latex(sp.Abs(sp.nsimplify(alpha_val, rational=False, tolerance=1e-8))
                               if not isinstance(alpha_val, sp.Basic)
                               else sp.Abs(sp.simplify(alpha_val)))

    # factor polinomial: vacío si k=0, n si k=1, n^k si k>1
    if k == 0:
        poly_latex = ""
    elif k == 1:
        poly_latex = r"n \cdot "
    else:
        poly_latex = rf"n^{{{k}}} \cdot "

    # armamos el cuerpo del término (sin el signo externo)
    cuerpo = rf"\left({alpha_latex}\right) \cdot {poly_latex}\left({r_latex}\right)^n"

    # el primer término no lleva '+' adelante, los siguientes sí (o '-' si es negativo)
    if es_primero:
        signo = "-" if negativo else ""
    else:
        signo = "-" if negativo else "+"

    return signo, cuerpo


# INTERFAZ STREAMLIT------------------------------------------

def main():
    st.title("Solución Cerrada de Recurrencia Lineal")
    st.markdown("### f(n) = a₁·f(n-1) + a₂·f(n-2) + ... + aₘ·f(n-m)")
    st.divider()

    # Leer m:
    # el grado m determina cuántos coeficientes y condiciones iniciales pedimos
    m = st.number_input("Grado m de la recurrencia:", min_value=1, step=1, value=2)

    st.divider()

    # Leer coeficientes:
    st.subheader("Coeficientes aᵢ")
    ar = []
    cols = st.columns(m)   # ponemos los inputs en columnas para que quede ordenado
    for i in range(m):
        with cols[i]:
            # cada coeficiente va en su columna con etiqueta clara
            a = st.number_input(f"a{i+1}", value=1, key=f"a{i}")
            ar.append(a)

    st.divider()

    # Leer condiciones iniciales:
    st.subheader("Condiciones iniciales Cᵢ")
    cr = []
    cols2 = st.columns(m)   # igual, columnas para los m valores iniciales
    for i in range(m):
        with cols2[i]:
            # valor por defecto es i para que fibonacci (0,1) funcione sin tocar nada
            c = st.number_input(f"C{i} = f({i})", value=int(i), key=f"c{i}")
            cr.append(c)

    st.divider()

    # Leer n:
    n_val = st.number_input("Valor de n a evaluar:", min_value=0, step=1, value=10)

    # Botón para calcular:
    if st.button("Calcular", type="primary"):

        # Validaciones de entrada:
        # antes de operar revisamos que los datos sean correctos
        try:
            validar_coeficientes(ar, m)
        except ValueError as e:
            st.error(f"Error en los coeficientes: {e}")
            return   # no tiene sentido continuar si los datos son inválidos

        try:
            validar_condiciones_iniciales(cr, m)
        except ValueError as e:
            st.error(f"Error en las condiciones iniciales: {e}")
            return

        # caso trivial: si n es menor que m, la respuesta es directamente un valor inicial
        if n_val < m:
            st.success(f"f({n_val}) = {cr[int(n_val)]}  (es una condición inicial directa)")
            return

        # Polinomio característico:
        # armamos [1, -a1, -a2, ..., -am] porque la ecuación es r^m - a1*r^(m-1) - ... = 0
        coefs_poly = [1] + [-a for a in ar]

        # Raíces numéricas (siempre las calculamos, las usamos para evaluar f(n)):
        try:
            raices_mult = obtener_raices_con_multiplicidad(coefs_poly)
        except np.linalg.LinAlgError as e:
            # numpy a veces falla internamente al calcular raíces de polinomios mal condicionados
            st.error(f"Error numérico al calcular raíces: {e}")
            return
        except Exception as e:
            # capturamos cualquier otra cosa inesperada
            st.error(f"Error inesperado al calcular raíces: {e}")
            return

        # verificamos que efectivamente salieron m raíces (contando multiplicidades)
        total_raices = sum(mult for (_, mult) in raices_mult)
        if total_raices != m:
            st.error(f"Se esperaban {m} raíces (con multiplicidad) pero se obtuvieron {total_raices}.")
            return

        # intentamos obtener raíces exactas con sympy para mostrar sqrt(5)/5 en lugar de 0.4472...
        raices_exactas_dict = obtener_raices_exactas(coefs_poly)
        modo_exacto = raices_exactas_dict is not None   # flag: True si sympy pudo factorizar

        # mostramos las raíces al usuario para que pueda revisar
        st.subheader("Raíces del polinomio característico")
        if modo_exacto:
            # mostramos en LaTeX exacto: p.ej. r = \frac{1 + \sqrt{5}}{2}
            for r_sp, mult in raices_exactas_dict.items():
                st.latex(rf"r = {sp.latex(r_sp)} \quad \text{{(multiplicidad {mult})}}")
        else:
            # si sympy no pudo, mostramos los valores numéricos como antes
            for (r, mult) in raices_mult:
                re = round(r.real, 6)
                im = round(r.imag, 6)
                if abs(im) < 1e-9:
                    # raíz real, no mostramos parte imaginaria para no confundir
                    st.write(f"r = {re}  (multiplicidad {mult})")
                else:
                    # raíz compleja, la mostramos completa
                    st.write(f"r = {re} + {im}i  (multiplicidad {mult})")

        # Sistema lineal numérico A·x = b:
        A = construir_sistema(raices_mult, m)        # matriz del sistema
        b = np.array(cr, dtype=complex)              # vector de condiciones iniciales

        try:
            det = np.linalg.det(A)
            # si el determinante es casi cero, el sistema no tiene solución única
            if abs(det) < 1e-12:
                raise np.linalg.LinAlgError("La matriz del sistema es singular (det ≈ 0).")
            coefs_solucion = np.linalg.solve(A, b)   # resolvemos A·x = b
        except np.linalg.LinAlgError as e:
            st.error(f"Error al resolver el sistema lineal: {e}")
            st.warning("Verifique que las condiciones iniciales sean consistentes.")
            return
        except Exception as e:
            # por si acaso hay algún otro error numérico que no esperábamos
            st.error(f"Error inesperado al resolver el sistema: {e}")
            return

        # verificamos que la solución tiene el tamaño correcto
        if len(coefs_solucion) != m:
            st.error("La solución del sistema no tiene el tamaño esperado.")
            return

        # si tenemos raíces exactas, intentamos también resolver los coeficientes exactos
        coefs_exactos = None
        if modo_exacto:
            coefs_exactos = resolver_coeficientes_exactos(raices_exactas_dict, cr)

        # Expresión cerrada:
        st.subheader("Expresión cerrada")
        terminos_latex = []   # vamos acumulando (signo, cuerpo_latex) de cada término

        if modo_exacto and coefs_exactos is not None:
            # modo exacto: raíces y coeficientes son expresiones simbólicas de sympy
            col_idx = 0
            for r_sp, mult in raices_exactas_dict.items():
                for k in range(mult):
                    alpha_sp = sp.simplify(coefs_exactos[col_idx])   # simplificamos antes de mostrar

                    # si el coeficiente es simbólicamente cero, lo saltamos
                    if alpha_sp == 0:
                        col_idx += 1
                        continue

                    es_primero = len(terminos_latex) == 0
                    signo, cuerpo = formatear_termino_latex(alpha_sp, r_sp, k, es_primero)
                    terminos_latex.append((signo, cuerpo))
                    col_idx += 1

        else:
            # modo numérico: usamos los coeficientes resueltos por numpy
            idx = 0
            for (r, mult) in raices_mult:
                for k in range(mult):
                    alpha_val = limpiar_alpha(coefs_solucion[idx])   # quitamos parte imaginaria si es cero

                    # si el coeficiente es prácticamente cero, no lo incluimos
                    if abs(alpha_val) < 1e-9:
                        idx += 1
                        continue

                    es_primero = len(terminos_latex) == 0
                    signo, cuerpo = formatear_termino_latex(alpha_val, r, k, es_primero)
                    terminos_latex.append((signo, cuerpo))
                    idx += 1

        # armamos el string LaTeX final juntando todos los términos con sus signos
        if terminos_latex:
            latex_str = "f(n) = "
            for signo, cuerpo in terminos_latex:
                # el primer término puede no tener signo (si es positivo)
                latex_str += f" {signo} {cuerpo}" if signo else cuerpo
            st.latex(latex_str)
        else:
            st.latex("f(n) = 0")

        # Evaluación y verificación:
        # evaluamos con la fórmula cerrada que encontramos
        resultado = evaluar_fn(raices_mult, coefs_solucion, int(n_val))

        # también calculamos iterativamente para verificar que cuadre
        f_iter = list(cr)   # empezamos con los valores iniciales
        for step in range(m, int(n_val) + 1):
            # aplicamos la recurrencia: f(n) = a1*f(n-1) + a2*f(n-2) + ...
            fn_paso = sum(ar[i] * f_iter[-(i + 1)] for i in range(m))
            f_iter.append(fn_paso)
        resultado_iter = round(f_iter[int(n_val)], 6)

        # mostramos ambos resultados lado a lado
        st.divider()
        st.subheader("Resultado")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"f({int(n_val)}) — Expresión cerrada", value=resultado)
        with col2:
            st.metric(label=f"f({int(n_val)}) — Verificación iterativa", value=resultado_iter)

        # comparamos los dos resultados para detectar errores numéricos
        if abs(resultado - resultado_iter) < 1e-3:
            st.success("Ambos métodos coinciden.")
        else:
            # si difieren mucho, probablemente hay inestabilidad numérica
            st.warning("Posible error numérico — los métodos difieren.")


if __name__ == '__main__':
    # Si Streamlit ya está corriendo, ejecuta la lógica del programa normalmente
    if runtime.exists():
        main()
    else:
        # Si no, se invoca a sí mismo a través de Streamlit
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(cli.main())
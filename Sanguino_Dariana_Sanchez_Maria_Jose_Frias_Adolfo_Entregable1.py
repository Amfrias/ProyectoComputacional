"""
Dado una recurrencia lineal homogénea de coeficientes constantes de grado m:

    f(n) = a1*f(n-1) + a2*f(n-2) + ... + am*f(n-m)

con condiciones iniciales f(0)=C0, f(1)=C1, ..., f(m-1)=Cm-1,
este programa encuentra la expresión cerrada (no recurrente) para f(n)
y la evalúa en un punto n dado por el usuario.

Interfaz gráfica construida con tkinter (incluido en Python estándar).
Las ecuaciones se renderizan con matplotlib (MathText / LaTeX subset).

Dependencias:
    - numpy      (raíces del polinomio y sistema lineal)
    - sympy      (manipulación simbólica)
    - matplotlib (renderizado de ecuaciones en la GUI)
    - tkinter    (GUI — incluido en Python)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mfig


# ─────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────

BG        = "#0f1117"   # fondo principal
SURFACE   = "#1a1d27"   # tarjetas / paneles
ACCENT    = "#7c6af7"   # violeta principal
ACCENT2   = "#a78bfa"   # violeta claro (hover / texto)
TEXT      = "#e8e6f0"   # texto primario
TEXT_MUTED= "#8b89a0"   # texto secundario
BORDER    = "#2e2a45"   # bordes sutiles
SUCCESS   = "#34d399"   # verde para coincidencia
ERROR     = "#f87171"   # rojo para error
INPUT_BG  = "#12141e"   # fondo de inputs


# ─────────────────────────────────────────────────────────────
# FUNCIONES MATEMÁTICAS (idénticas a la versión consola)
# ─────────────────────────────────────────────────────────────

def obtener_raices_con_multiplicidad(coefs_poly, tol=1e-6):
    """
    Calcula las raíces del polinomio característico y las agrupa
    por multiplicidad usando tolerancia numérica.

    Parámetros:
        coefs_poly : lista [1, -a1, -a2, ..., -am]
        tol        : tolerancia para considerar dos raíces iguales

    Retorna:
        lista de tuplas (raiz_compleja, multiplicidad)
    """
    raices_raw = np.roots(coefs_poly)
    agrupadas  = []
    usada      = [False] * len(raices_raw)

    for i, r in enumerate(raices_raw):
        if usada[i]:
            continue
        mult     = 1
        usada[i] = True
        for j in range(i + 1, len(raices_raw)):
            if not usada[j] and abs(r - raices_raw[j]) < tol:
                mult       += 1
                usada[j]    = True
        agrupadas.append((r, mult))

    return agrupadas


def construir_sistema(raices_mult, m):
    """
    Arma la matriz A del sistema A·x = b imponiendo las m condiciones
    iniciales a la solución general.

    Cada raíz r con multiplicidad k aporta columnas: r^n, n·r^n, …, n^(k-1)·r^n.

    Parámetros:
        raices_mult : lista de (raiz, multiplicidad)
        m           : grado de la recurrencia

    Retorna:
        A : numpy array (m x m) compleja
    """
    A   = np.zeros((m, m), dtype=complex)
    col = 0
    for (r, mult) in raices_mult:
        for k in range(mult):
            for fila in range(m):
                n     = fila
                pot_n = 1 if (n == 0 and k == 0) else n ** k
                pot_r = 1 if n == 0 else r ** n
                A[fila, col] = pot_n * pot_r
            col += 1
    return A


def evaluar_fn(raices_mult, coefs_solucion, n_val):
    """
    Evalúa numéricamente f(n) usando la expresión cerrada.

    Parámetros:
        raices_mult    : lista de (raiz, multiplicidad)
        coefs_solucion : coeficientes α resueltos
        n_val          : entero n a evaluar

    Retorna:
        parte real de f(n_val) redondeada a 6 decimales
    """
    resultado = 0 + 0j
    idx       = 0
    for (r, mult) in raices_mult:
        for k in range(mult):
            alpha  = coefs_solucion[idx]
            n_k    = 1 if (n_val == 0 and k == 0) else n_val ** k
            r_n    = 1 if n_val == 0 else r ** n_val
            resultado += alpha * n_k * r_n
            idx   += 1
    return round(resultado.real, 6)


def resolver_recurrencia(m, ar, cr, n_val):
    """
    Orquesta todo el cálculo:
      1. Construye el polinomio característico.
      2. Halla raíces con multiplicidad.
      3. Resuelve el sistema para los coeficientes α.
      4. Evalúa f(n) por expresión cerrada y verificación iterativa.

    Parámetros:
        m     : grado (int)
        ar    : lista de coeficientes [a1, …, am]
        cr    : condiciones iniciales [C0, …, Cm-1]
        n_val : valor de n a evaluar

    Retorna:
        dict con claves: raices_mult, coefs_solucion,
                         resultado_cerrado, resultado_iter, coinciden
    """
    # Caso trivial: n es condición inicial directa
    if n_val < m:
        return {"directo": True, "valor": cr[n_val]}

    # Polinomio característico: r^m - a1·r^(m-1) - … - am = 0
    coefs_poly  = [1] + [-a for a in ar]
    raices_mult = obtener_raices_con_multiplicidad(coefs_poly)

    # Sistema lineal A·α = b
    A = construir_sistema(raices_mult, m)
    b = np.array(cr, dtype=complex)

    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise np.linalg.LinAlgError("La matriz del sistema es singular (det ≈ 0).")

    coefs_solucion = np.linalg.solve(A, b)

    # Evaluar con expresión cerrada
    resultado_cerrado = evaluar_fn(raices_mult, coefs_solucion, n_val)

    # Verificación iterativa
    f_iter = list(cr)
    for step in range(m, n_val + 1):
        fn_paso = sum(ar[i] * f_iter[-(i + 1)] for i in range(m))
        f_iter.append(fn_paso)
    resultado_iter = round(f_iter[n_val], 6)

    coinciden = abs(resultado_cerrado - resultado_iter) < 1e-3

    return {
        "directo"          : False,
        "raices_mult"      : raices_mult,
        "coefs_solucion"   : coefs_solucion,
        "resultado_cerrado": resultado_cerrado,
        "resultado_iter"   : resultado_iter,
        "coinciden"        : coinciden,
    }


# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE STRINGS LATEX PARA LAS ECUACIONES
# ─────────────────────────────────────────────────────────────

def latex_recurrencia(ar):
    """
    Genera el string LaTeX de la recurrencia f(n) = a1·f(n-1) + …

    Parámetros:
        ar : lista de coeficientes [a1, …, am]

    Retorna:
        string LaTeX (sin $…$)
    """
    partes = []
    for i, a in enumerate(ar):
        a_fmt = _fmt_coef(a)
        if i == 0:
            partes.append(rf"{a_fmt}\,f(n-{i+1})")
        else:
            signo = "+" if a >= 0 else ""
            partes.append(rf"{signo}{a_fmt}\,f(n-{i+1})")
    return r"f(n) = " + " ".join(partes)


def latex_expresion_cerrada(raices_mult, coefs_solucion):
    """
    Construye el string LaTeX de la solución general f(n) = Σ αk·n^j·r^n.

    Parámetros:
        raices_mult    : lista de (raiz, multiplicidad)
        coefs_solucion : coeficientes α

    Retorna:
        string LaTeX (sin $…$)
    """
    terminos = []
    idx      = 0

    for (r, mult) in raices_mult:
        for k in range(mult):
            alpha = coefs_solucion[idx]
            re    = round(alpha.real, 6)
            im    = round(alpha.imag, 6)
            idx  += 1

            if abs(complex(re, im)) < 1e-9:
                continue

            # Coeficiente α
            if abs(im) < 1e-6:
                coef_str = _fmt_coef(re)
            else:
                coef_str = rf"({re:.4g}{'+' if im >= 0 else ''}{im:.4g}i)"

            # Factor polinomial n^k
            if k == 0:
                poly_str = ""
            elif k == 1:
                poly_str = r"n \cdot "
            else:
                poly_str = rf"n^{{{k}}} \cdot "

            # Base r^n
            r_re = round(r.real, 6)
            r_im = round(r.imag, 6)
            if abs(r_im) < 1e-6:
                base_str = rf"({r_re:.4g})^n"
            else:
                base_str = rf"({r_re:.4g}{'+' if r_im >= 0 else ''}{r_im:.4g}i)^n"

            terminos.append(rf"{coef_str} \cdot {poly_str}{base_str}")

    if not terminos:
        return r"f(n) = 0"

    latex = r"f(n) = " + r" \;+\; ".join(terminos)
    return latex


def _fmt_coef(v):
    """
    Formatea un número como entero si es entero, o decimal con 4 cifras sig.

    Parámetros:
        v : float

    Retorna:
        string representando el número
    """
    if abs(v - round(v)) < 1e-8:
        return str(int(round(v)))
    return f"{v:.4g}"


# ─────────────────────────────────────────────────────────────
# WIDGET REUTILIZABLE: RENDER DE ECUACIÓN CON MATPLOTLIB
# ─────────────────────────────────────────────────────────────

class EcuacionWidget(tk.Frame):
    """
    Frame que embebe un canvas de matplotlib para renderizar
    una ecuación LaTeX de forma legible dentro de tkinter.
    """

    def __init__(self, parent, height=60, **kwargs):
        super().__init__(parent, bg=SURFACE, **kwargs)
        self.height = height
        # Figura matplotlib transparente
        self.fig = mfig.Figure(figsize=(7, height / 100), dpi=100)
        self.fig.patch.set_facecolor(SURFACE)
        self.ax  = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(SURFACE)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def mostrar(self, latex_str, fontsize=13):
        """
        Renderiza el string LaTeX en el canvas.

        Parámetros:
            latex_str : string LaTeX (sin $…$)
            fontsize  : tamaño de fuente (default 13)
        """
        self.ax.clear()
        self.ax.set_facecolor(SURFACE)
        self.ax.axis("off")
        try:
            self.ax.text(
                0.5, 0.5,
                f"${latex_str}$",
                ha="center", va="center",
                fontsize=fontsize,
                color=TEXT,
                transform=self.ax.transAxes,
                usetex=False,   # usa MathText de matplotlib, sin LaTeX externo
            )
        except Exception:
            # Fallback a texto plano si falla el parseo
            self.ax.text(0.5, 0.5, latex_str,
                         ha="center", va="center",
                         fontsize=fontsize - 2, color=TEXT_MUTED,
                         transform=self.ax.transAxes)
        self.canvas.draw()

    def limpiar(self):
        """Borra el contenido del canvas."""
        self.ax.clear()
        self.ax.set_facecolor(SURFACE)
        self.ax.axis("off")
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────
# VENTANA PRINCIPAL
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    """
    Aplicación principal. Organiza:
      - Panel izquierdo: entradas (m, coeficientes, condiciones, n)
      - Panel derecho:   resultados (raíces, expresión cerrada, f(n))
    """

    def __init__(self):
        super().__init__()
        self.title("Recurrencia Lineal Homogénea")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 580)

        # Estado dinámico
        self.m_var     = tk.IntVar(value=2)
        self.coef_vars = []   # StringVars para a1…am
        self.ci_vars   = []   # StringVars para C0…Cm-1
        self.n_var     = tk.StringVar(value="10")

        self._construir_ui()
        self._actualizar_entradas(None)   # inicializa campos para m=2

    # ── CONSTRUCCIÓN DE LA UI ─────────────────────────────────

    def _construir_ui(self):
        """Arma la estructura de dos paneles de la ventana."""

        # Título superior
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(header, text="Solución cerrada de recurrencias",
                 bg=BG, fg=ACCENT2, font=("Helvetica", 17, "bold")).pack(side="left")

        tk.Label(header,
                 text="f(n) = a₁·f(n−1) + a₂·f(n−2) + … + aₘ·f(n−m)",
                 bg=BG, fg=TEXT_MUTED, font=("Helvetica", 11)).pack(side="left", padx=16)

        # Separador
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=4)

        # Contenedor de dos columnas
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=12)

        # Panel izquierdo (entradas)
        self.panel_izq = tk.Frame(main, bg=SURFACE,
                                  highlightbackground=BORDER,
                                  highlightthickness=1)
        self.panel_izq.pack(side="left", fill="both", expand=False,
                            padx=(0, 12), pady=0, ipadx=16, ipady=16)
        self.panel_izq.pack_propagate(True)

        # Panel derecho (resultados)
        self.panel_der = tk.Frame(main, bg=SURFACE,
                                  highlightbackground=BORDER,
                                  highlightthickness=1)
        self.panel_der.pack(side="left", fill="both", expand=True,
                            ipadx=16, ipady=16)

        self._construir_panel_entrada()
        self._construir_panel_resultado()

    def _lbl(self, parent, text, muted=False, size=11):
        """Helper: crea un Label con el estilo de la app."""
        return tk.Label(parent, text=text, bg=SURFACE,
                        fg=TEXT_MUTED if muted else TEXT,
                        font=("Helvetica", size))

    def _entry(self, parent, textvariable, width=8):
        """Helper: crea un Entry con el estilo oscuro."""
        e = tk.Entry(parent, textvariable=textvariable,
                     bg=INPUT_BG, fg=TEXT, insertbackground=ACCENT2,
                     relief="flat", font=("Courier", 11),
                     width=width,
                     highlightthickness=1,
                     highlightbackground=BORDER,
                     highlightcolor=ACCENT)
        return e

    # ── PANEL IZQUIERDO ───────────────────────────────────────

    def _construir_panel_entrada(self):
        """Construye el panel de entradas (grado m, coeficientes, CIs, n)."""
        p = self.panel_izq

        # Grado m
        f_m = tk.Frame(p, bg=SURFACE)
        f_m.pack(fill="x", pady=(0, 14))
        self._lbl(f_m, "Grado  m", size=12).pack(side="left")
        spin = tk.Spinbox(f_m, from_=1, to=8,
                          textvariable=self.m_var,
                          width=4, bg=INPUT_BG, fg=ACCENT2,
                          buttonbackground=SURFACE,
                          relief="flat", font=("Courier", 12),
                          highlightthickness=1,
                          highlightbackground=BORDER,
                          highlightcolor=ACCENT,
                          command=lambda: self._actualizar_entradas(None))
        spin.pack(side="left", padx=10)
        spin.bind("<Return>", self._actualizar_entradas)
        spin.bind("<FocusOut>", self._actualizar_entradas)

        # Contenedor dinámico para coeficientes y CIs
        self.frame_dinamico = tk.Frame(p, bg=SURFACE)
        self.frame_dinamico.pack(fill="x")

        # Separador antes de n
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", pady=10)

        # Valor de n
        f_n = tk.Frame(p, bg=SURFACE)
        f_n.pack(fill="x", pady=(0, 14))
        self._lbl(f_n, "Evaluar en  n =", size=12).pack(side="left")
        self._entry(f_n, self.n_var, width=6).pack(side="left", padx=10)

        # Botón calcular
        btn = tk.Button(p, text="  Calcular  →",
                        command=self._calcular,
                        bg=ACCENT, fg="#ffffff",
                        activebackground=ACCENT2,
                        activeforeground="#ffffff",
                        relief="flat", font=("Helvetica", 12, "bold"),
                        cursor="hand2", pady=8)
        btn.pack(fill="x", pady=(8, 0))

    def _actualizar_entradas(self, event):
        """
        Regenera dinámicamente los campos de coeficientes y condiciones
        iniciales cada vez que cambia el grado m.
        """
        try:
            m = int(self.m_var.get())
            if m < 1:
                return
        except (ValueError, tk.TclError):
            return

        # Destruir widgets anteriores
        for w in self.frame_dinamico.winfo_children():
            w.destroy()

        # Preservar valores previos si existen
        prev_coefs = [v.get() for v in self.coef_vars]
        prev_cis   = [v.get() for v in self.ci_vars]

        self.coef_vars = []
        self.ci_vars   = []

        # ── Sección coeficientes ──
        self._lbl(self.frame_dinamico,
                  "Coeficientes  a₁ … aₘ", muted=True).pack(anchor="w", pady=(0, 4))

        for i in range(m):
            row = tk.Frame(self.frame_dinamico, bg=SURFACE)
            row.pack(fill="x", pady=2)
            self._lbl(row, f"a{i+1}", size=11).pack(side="left", padx=(0, 6))
            var = tk.StringVar(value=prev_coefs[i] if i < len(prev_coefs) else "")
            self._entry(row, var).pack(side="left")
            self.coef_vars.append(var)

        # ── Sección condiciones iniciales ──
        tk.Frame(self.frame_dinamico, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._lbl(self.frame_dinamico,
                  "Condiciones iniciales  f(0) … f(m−1)", muted=True).pack(anchor="w", pady=(0, 4))

        for i in range(m):
            row = tk.Frame(self.frame_dinamico, bg=SURFACE)
            row.pack(fill="x", pady=2)
            self._lbl(row, f"f({i})", size=11).pack(side="left", padx=(0, 6))
            var = tk.StringVar(value=prev_cis[i] if i < len(prev_cis) else "")
            self._entry(row, var).pack(side="left")
            self.ci_vars.append(var)

    # ── PANEL DERECHO ─────────────────────────────────────────

    def _construir_panel_resultado(self):
        """Construye el panel de resultados con placeholders."""
        p = self.panel_der

        self._lbl(p, "Recurrencia ingresada", muted=True, size=10).pack(anchor="w")
        self.eq_recurrencia = EcuacionWidget(p, height=55)
        self.eq_recurrencia.pack(fill="x", pady=(2, 10))

        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", pady=4)

        self._lbl(p, "Raíces del polinomio característico",
                  muted=True, size=10).pack(anchor="w")
        self.txt_raices = tk.Text(p, bg=INPUT_BG, fg=TEXT_MUTED,
                                  relief="flat", font=("Courier", 10),
                                  height=4, state="disabled",
                                  highlightthickness=0)
        self.txt_raices.pack(fill="x", pady=(2, 10))

        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", pady=4)

        self._lbl(p, "Expresión cerrada", muted=True, size=10).pack(anchor="w")
        self.eq_cerrada = EcuacionWidget(p, height=65)
        self.eq_cerrada.pack(fill="x", pady=(2, 12))

        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", pady=4)

        # Tarjetas de resultado
        cards = tk.Frame(p, bg=SURFACE)
        cards.pack(fill="x", pady=(4, 0))

        self.card_cerrado = self._tarjeta(cards, "Expresión cerrada", "—")
        self.card_cerrado.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.card_iter = self._tarjeta(cards, "Verificación iterativa", "—")
        self.card_iter.pack(side="left", fill="both", expand=True)

        # Indicador de coincidencia
        self.lbl_coincide = tk.Label(p, text="", bg=SURFACE,
                                     font=("Helvetica", 11, "bold"))
        self.lbl_coincide.pack(pady=(10, 0))

    def _tarjeta(self, parent, titulo, valor):
        """
        Crea una tarjeta de métrica (título + valor grande).

        Parámetros:
            parent : widget padre
            titulo : string del título
            valor  : valor inicial a mostrar

        Retorna:
            Frame de la tarjeta (con atributo .lbl_valor para actualizarlo)
        """
        frame = tk.Frame(parent, bg=INPUT_BG,
                         highlightbackground=BORDER,
                         highlightthickness=1)
        tk.Label(frame, text=titulo, bg=INPUT_BG,
                 fg=TEXT_MUTED, font=("Helvetica", 10)).pack(pady=(10, 2))
        lbl = tk.Label(frame, text=valor, bg=INPUT_BG,
                       fg=ACCENT2, font=("Helvetica", 20, "bold"))
        lbl.pack(pady=(0, 10))
        frame.lbl_valor = lbl   # referencia para actualizar
        return frame

    # ── LÓGICA DE CÁLCULO ─────────────────────────────────────

    def _calcular(self):
        """
        Lee las entradas, valida, llama a resolver_recurrencia()
        y actualiza todos los widgets del panel derecho.
        """
        # ── Leer y validar m ──
        try:
            m = int(self.m_var.get())
            assert m >= 1
        except Exception:
            messagebox.showerror("Error", "El grado m debe ser un entero ≥ 1.")
            return

        # ── Leer coeficientes ──
        ar = []
        for i, var in enumerate(self.coef_vars):
            try:
                ar.append(float(var.get()))
            except ValueError:
                messagebox.showerror("Error", f"Coeficiente a{i+1} no es válido.")
                return

        # ── Leer condiciones iniciales ──
        cr = []
        for i, var in enumerate(self.ci_vars):
            try:
                cr.append(float(var.get()))
            except ValueError:
                messagebox.showerror("Error", f"Condición inicial f({i}) no es válida.")
                return

        # ── Leer n ──
        try:
            n_val = int(self.n_var.get())
            assert n_val >= 0
        except Exception:
            messagebox.showerror("Error", "n debe ser un entero ≥ 0.")
            return

        # ── Actualizar ecuación de recurrencia ──
        self.eq_recurrencia.mostrar(latex_recurrencia(ar), fontsize=12)

        # ── Resolver ──
        try:
            res = resolver_recurrencia(m, ar, cr, n_val)
        except np.linalg.LinAlgError as e:
            messagebox.showerror("Error numérico", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # ── Caso directo (n es condición inicial) ──
        if res.get("directo"):
            self.txt_raices.config(state="normal")
            self.txt_raices.delete("1.0", "end")
            self.txt_raices.insert("end", f"n={n_val} es condición inicial directa.")
            self.txt_raices.config(state="disabled")
            self.eq_cerrada.limpiar()
            v = res["valor"]
            self.card_cerrado.lbl_valor.config(text=str(v), fg=SUCCESS)
            self.card_iter.lbl_valor.config(text=str(v), fg=SUCCESS)
            self.lbl_coincide.config(text=f"f({n_val}) = {v}  (condición inicial)",
                                     fg=SUCCESS)
            return

        # ── Mostrar raíces ──
        self.txt_raices.config(state="normal")
        self.txt_raices.delete("1.0", "end")
        for (r, mult) in res["raices_mult"]:
            re = round(r.real, 6)
            im = round(r.imag, 6)
            if abs(im) < 1e-9:
                linea = f"  r = {re}   (multiplicidad {mult})\n"
            else:
                signo = "+" if im >= 0 else ""
                linea = f"  r = {re} {signo} {im}i   (multiplicidad {mult})\n"
            self.txt_raices.insert("end", linea)
        self.txt_raices.config(state="disabled")

        # ── Mostrar expresión cerrada ──
        latex_ec = latex_expresion_cerrada(
            res["raices_mult"], res["coefs_solucion"])
        self.eq_cerrada.mostrar(latex_ec, fontsize=12)

        # ── Actualizar tarjetas ──
        vc = res["resultado_cerrado"]
        vi = res["resultado_iter"]
        color_vc = SUCCESS if res["coinciden"] else ERROR
        color_vi = SUCCESS if res["coinciden"] else ERROR

        self.card_cerrado.lbl_valor.config(text=str(vc), fg=color_vc)
        self.card_iter.lbl_valor.config(text=str(vi), fg=color_vi)

        # ── Indicador final ──
        if res["coinciden"]:
            self.lbl_coincide.config(
                text=f"✓  f({n_val}) = {vc}  — ambos métodos coinciden",
                fg=SUCCESS)
        else:
            self.lbl_coincide.config(
                text=f"⚠  posible error numérico  |  cerrada: {vc}  iter: {vi}",
                fg=ERROR)


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
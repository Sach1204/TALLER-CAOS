
"""Taller computacional: caos, conjunto de Mandelbrot y universalidad de Feigenbaum."""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "figs")
os.makedirs(OUT, exist_ok=True)
RES = {}
DELTA = 4.669201609

# =====================================================================
# PUNTO 1: órbitas complejas
# =====================================================================
def orbita(c, nmax=100, radio=2.0, nprim=8):
    z = 0j
    orb = [z]
    max_mod = 0.0
    for n in range(1, nmax + 1):
        z = z * z + c
        orb.append(z)
        max_mod = max(max_mod, abs(z))
        if abs(z) > radio:
            return dict(escapo=True, n_escape=n, primeros=orb[:nprim], max_mod=max_mod)
    return dict(escapo=False, n_escape=None, primeros=orb[:nprim], max_mod=max_mod)


def periodo_orbita(c, transit=20000, maxp=64, tol=1e-9):
    z = 0j
    for _ in range(transit):
        z = z * z + c
        if abs(z) > 2:
            return None
    ref = z
    for p in range(1, maxp + 1):
        z = z * z + c
        if abs(z - ref) < tol:
            return p
    return 0  # acotada sin ciclo detectado


cs = [0, -1, 1, -0.75 + 0.1j, -0.1 + 0.65j]
p1 = []
for c in cs:
    o = orbita(c, 100)
    o_largo = orbita(c, 100000)
    per = periodo_orbita(c)
    p1.append(dict(
        c=str(c), escapo=o["escapo"], n_escape=o["n_escape"], max_mod=o["max_mod"],
        primeros=[f"{z.real:.4f}{z.imag:+.4f}i" for z in o["primeros"]],
        escapo_100000=o_largo["escapo"], n_escape_100000=o_largo["n_escape"],
        max_mod_100000=o_largo["max_mod"], periodo=per))
RES["punto1"] = p1

# =====================================================================
# PUNTO 2 y 3: Mandelbrot
# =====================================================================
def mandelbrot(xmin, xmax, ymin, ymax, nx, ny, nmax, radio=2.0):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C)
    N = np.zeros(C.shape, dtype=np.int32)
    vivo = np.ones(C.shape, dtype=bool)
    for k in range(1, nmax + 1):
        Z[vivo] = Z[vivo] ** 2 + C[vivo]
        esc = vivo & (np.abs(Z) > radio)
        N[esc] = k
        vivo &= ~esc
    return np.ma.masked_where(vivo, N), vivo


def dibujar_mandel(N, ext, titulo, fname, nmax):
    fig, ax = plt.subplots(figsize=(7, 6.4))
    cmap = plt.get_cmap("twilight_shifted").copy()
    cmap.set_bad("black")
    im = ax.imshow(N, extent=ext, origin="lower", cmap=cmap, aspect="equal",
                   vmin=1, vmax=nmax, interpolation="nearest")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_title(titulo)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("iteración de escape")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=110)
    plt.close(fig)


ext = (-2.0, 1.0, -1.5, 1.5)
p2 = []
for nmax in [50, 100, 200, 500]:
    t0 = time.perf_counter()
    N, vivo = mandelbrot(*ext, 800, 800, nmax)
    dt = time.perf_counter() - t0
    frac = vivo.mean()
    area = frac * 3.0 * 3.0
    p2.append(dict(nmax=nmax, tiempo=dt, frac_interior=float(frac), area=float(area)))
    dibujar_mandel(N, ext, f"Conjunto de Mandelbrot (Nmax = {nmax}, 800×800)",
                   f"mandel_{nmax}.png", nmax)
RES["punto2"] = p2

# Ampliaciones sucesivas (zona del "valle de los caballitos de mar")
ventanas = [
    ((-0.80, -0.70, 0.05, 0.15), 800, 300),
    ((-0.755, -0.735, 0.103, 0.123), 800, 600),
    ((-0.7465, -0.7445, 0.1115, 0.1135), 800, 1200),
]
p3 = []
for i, (v, res, nmax) in enumerate(ventanas, 1):
    t0 = time.perf_counter()
    N, vivo = mandelbrot(v[0], v[1], v[2], v[3], res, res, nmax)
    dt = time.perf_counter() - t0
    p3.append(dict(i=i, lim=v, res=res, nmax=nmax, tiempo=dt, frac=float(vivo.mean())))
    dibujar_mandel(N, v, f"Ampliación {i}: Re ∈ [{v[0]}, {v[1]}], Im ∈ [{v[2]}, {v[3]}]",
                   f"zoom_{i}.png", nmax)
# Experimento de tiempo vs resolución y Nmax (ventana 1)
tiempos = []
for res in [200, 400, 800]:
    for nmax in [100, 300, 1000]:
        t0 = time.perf_counter()
        mandelbrot(*ventanas[0][0], res, res, nmax)
        tiempos.append(dict(res=res, nmax=nmax, t=time.perf_counter() - t0))
RES["punto3"] = p3
RES["tiempos"] = tiempos

# =====================================================================
# Mapa logístico: utilidades
# =====================================================================
def fpn(x, r, p):
    """f^p(x) y derivada (f^p)'(x) para el mapa logístico."""
    d = 1.0
    for _ in range(p):
        d *= r * (1 - 2 * x)
        x = r * x * (1 - x)
    return x, d

# Parámetros superestables R_k (ciclo de periodo 2^k que contiene x = 1/2)
Rs = [2.0, 1 + np.sqrt(5)]
for k in range(2, 10):
    guess = Rs[-1] + (Rs[-1] - Rs[-2]) / 4.669
    sol = fsolve(lambda r: fpn(0.5, r[0], 2 ** k)[0] - 0.5, [guess], xtol=1e-14)[0]
    Rs.append(sol)

# Puntos de bifurcación r_n (multiplicador = -1 del ciclo de periodo 2^(n-1))
rb = [3.0]
for n in range(2, 10):
    p = 2 ** (n - 1)
    r0 = Rs[n - 1] + 0.8 * (Rs[n] - Rs[n - 1])
    def G(v, p=p):
        x, r = v
        val, d = fpn(x, r, p)
        return [val - x, d + 1.0]
    sol = fsolve(G, [0.5, r0], xtol=1e-14)
    rb.append(sol[1])
rb = np.array(rb)

# =====================================================================
# PUNTO 4: diagrama de bifurcación
# =====================================================================
def bif(rmin, rmax, nr=6000, niter=1500, descartar=1000):
    r = np.linspace(rmin, rmax, nr)
    x = 0.5 * np.ones(nr)
    X = np.empty((niter - descartar, nr))
    for i in range(niter):
        x = r * x * (1 - x)
        if i >= descartar:
            X[i - descartar] = x
    return r, X

t0 = time.perf_counter()
r_full, X_full = bif(2.5, 4.0)
t_bif = time.perf_counter() - t0
r_zoom, X_zoom = bif(3.4, 3.6)

def dibujar_bif(r, X, fname, titulo, xlim, marcas=None, textos=None, size=(11, 6)):
    fig, ax = plt.subplots(figsize=size)
    ax.plot(np.tile(r, X.shape[0]), X.ravel(), ",", color="k", alpha=0.35)
    ax.set_xlim(*xlim)
    ax.set_xlabel("r")
    ax.set_ylabel("$x_n$ (últimos 500 valores)")
    ax.set_title(titulo, pad=22)
    if marcas:
        for m, lab in marcas:
            ax.axvline(m, color="tab:red", lw=0.7, ls="--", alpha=0.8)
            ax.text(m, 1.02, lab, color="tab:red", ha="center", va="bottom", fontsize=8,
                    transform=ax.get_xaxis_transform())
    if textos:
        for (x, y, s) in textos:
            ax.text(x, y, s, fontsize=9, bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=1.5))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=120)
    plt.close(fig)

marc = [(rb[0], "$r_1$"), (rb[1], "$r_2$"), (3.5699456, "$r_\\infty$"), (3.8284, "T3")]
tex = [(2.55, 0.62, "punto fijo"), (3.07, 0.25, "periodo 2"), (3.46, 0.15, "periodo 4"),
       (3.62, 0.60, "caos"), (3.72, 0.05, "ventana periodo 3 →")]
dibujar_bif(r_full, X_full, "bif_completo.png",
            "Diagrama de bifurcación del mapa logístico (6000 valores de r, $x_0$=0.5)",
            (2.5, 4.0), marc, tex)
dibujar_bif(r_zoom, X_zoom, "bif_zoom.png",
            "Ampliación 3.4 ≤ r ≤ 3.6: periodos 4, 8, 16 y entrada al caos",
            (3.4, 3.6), [(rb[2], "$r_3$ (8)"), (rb[3], "$r_4$ (16)"), (rb[4], "$r_5$ (32)")], None)

# Ventana de periodo 3 (ampliación adicional)
r_w, X_w = bif(3.8, 3.87, nr=4000, niter=2500, descartar=2000)
dibujar_bif(r_w, X_w, "bif_ventana3.png", "Ventana periódica de periodo 3 (3.80 ≤ r ≤ 3.87)",
            (3.8, 3.87), None, None, size=(8, 5))

# Transitorio: series temporales
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, r in zip(axs, [3.2, 3.5]):
    x = 0.5
    xs = []
    for _ in range(80):
        x = r * x * (1 - x)
        xs.append(x)
    ax.plot(xs, "o-", ms=3, lw=0.8)
    ax.set_title(f"r = {r}")
    ax.set_xlabel("n")
axs[0].set_ylabel("$x_n$")
fig.suptitle("Primeras 80 iteraciones desde $x_0$ = 0.5: se ve el transitorio antes del ciclo")
fig.tight_layout()
fig.savefig(f"{OUT}/transitorio.png", dpi=110)
plt.close(fig)

# Estimación "gráfica": primer r del diagrama con >= 2^n valores distintos
def estimacion_grafica(r, X, dec=3):
    conteos = np.array([len(np.unique(np.round(X[:, j], dec))) for j in range(X.shape[1])])
    out = []
    for n in range(1, 7):
        idx = np.where(conteos >= 2 ** n)[0]
        out.append(float(r[idx[0]]) if len(idx) else None)
    return out
graf = estimacion_grafica(r_full, X_full)

# =====================================================================
# PUNTO 5: constante de Feigenbaum
# =====================================================================
# (a) búsqueda progresiva sobre detección de periodo
def pred_no_periodo(rs, p, T, tol=1e-6):
    x = 0.5 * np.ones_like(rs)
    for _ in range(T):
        x = rs * x * (1 - x)
    y = x.copy()
    for _ in range(p):
        y = rs * y * (1 - y)
    return np.abs(y - x) > tol

def busqueda_progresiva(n, T, ngrid=400, niveles=4):
    lo, hi = Rs[n - 1], Rs[n]
    p = 2 ** (n - 1)
    for _ in range(niveles):
        g = np.linspace(lo, hi, ngrid)
        m = pred_no_periodo(g, p, T)
        i = int(np.argmax(m))
        lo, hi = g[max(i - 1, 0)], g[i]
    return 0.5 * (lo + hi)

busq = {}
tb0 = time.perf_counter()
for T in [1000, 10000, 100000]:
    busq[T] = [busqueda_progresiva(n, T) for n in range(1, 8)]
t_busq = time.perf_counter() - tb0

def deltas(r):
    return [(r[i] - r[i - 1]) / (r[i + 1] - r[i]) for i in range(1, len(r) - 1)]

d_ref = deltas(list(rb))  # d_ref[j] corresponde a n = j+2
E_ref = [abs(d - DELTA) / DELTA * 100 for d in d_ref]
r_inf = rb[-1] + (rb[-1] - rb[-2]) / (DELTA - 1)

tabla5 = []
for i in range(len(rb)):
    fila = dict(n=i + 1, periodo=2 ** (i + 1), r=float(rb[i]),
                dr=float(rb[i] - rb[i - 1]) if i > 0 else None,
                delta=None, err=None)
    if 1 <= i < len(rb) - 1:
        fila["delta"] = d_ref[i - 1]
        fila["err"] = E_ref[i - 1]
    tabla5.append(fila)

RES["punto5"] = dict(
    tabla=tabla5, rb=[float(v) for v in rb], Rs=[float(v) for v in Rs], r_inf=float(r_inf),
    graf=graf, t_busq=t_busq,
    busq={str(T): [float(v) for v in vals] for T, vals in busq.items()},
    busq_err={str(T): [float(abs(a - b)) for a, b in zip(vals, rb[:7])] for T, vals in busq.items()},
    busq_delta={str(T): deltas(vals) for T, vals in busq.items()},
)
RES["t_bif"] = t_bif

# Figura: delta_n
fig, ax = plt.subplots(figsize=(7, 4))
ns = np.arange(2, 2 + len(d_ref))
ax.plot(ns, d_ref, "o-", label="$\\delta_n$ (refinado, multiplicador = −1)")
dg = deltas(busq[100000])
ax.plot(np.arange(2, 2 + len(dg)), dg, "s--", alpha=0.7, label="$\\delta_n$ (búsqueda progresiva, T=10$^5$)")
ax.axhline(DELTA, color="tab:red", ls=":", label="δ = 4.669201609")
ax.set_xlabel("n")
ax.set_ylabel("$\\delta_n$")
ax.set_title("Convergencia de las razones de Feigenbaum")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/delta.png", dpi=120)
plt.close(fig)

# =====================================================================
# RETO: Mandelbrot en el eje real vs mapa logístico
# =====================================================================
def periodo_logistico(r, T=200000, maxp=64, tol=1e-7):
    x = 0.5
    for _ in range(T):
        x = r * x * (1 - x)
    ref = x
    for p in range(1, maxp + 1):
        x = r * x * (1 - x)
        if abs(x - ref) < tol:
            return p
    return 0

def periodo_cuadratico(c, T=200000, maxp=64, tol=1e-7):
    z = 0.0
    for _ in range(T):
        z = z * z + c
    ref = z
    for p in range(1, maxp + 1):
        z = z * z + c
        if abs(z - ref) < tol:
            return p
    return 0

reto = []
for r in [2.8, 3.2, 3.5, 3.56, 3.83, 3.9, 3.99]:
    c = r / 2 * (1 - r / 2)
    reto.append(dict(r=r, c=c, per_x=periodo_logistico(r), per_z=periodo_cuadratico(c)))
# conjugación explícita: x = 1/2 - z/r
r = 3.9
c = r / 2 * (1 - r / 2)
x = 0.3
z = r * (0.5 - x)
err_conj = 0.0
for _ in range(50):
    x = r * x * (1 - x)
    z = z * z + c
    err_conj = max(err_conj, abs(z - r * (0.5 - x)))
RES["reto"] = dict(tabla=reto, err_conj=err_conj,
                   c_bif=[float(v / 2 * (1 - v / 2)) for v in rb],
                   c_inf=float(r_inf / 2 * (1 - r_inf / 2)))

# Figura: diagrama en c
cc = np.linspace(0.25, -2.0, 6000)
z = np.zeros_like(cc)
Zc = np.empty((500, cc.size))
for i in range(1500):
    z = z * z + cc
    if i >= 1000:
        Zc[i - 1000] = z
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
axs[0].plot(np.tile(r_full, 500), X_full.ravel(), ",", color="k", alpha=0.35)
axs[0].set_xlabel("r"); axs[0].set_ylabel("$x_n$"); axs[0].set_title("Mapa logístico")
axs[1].plot(np.tile(cc, 500), Zc.ravel(), ",", color="tab:blue", alpha=0.35)
axs[1].set_xlim(0.25, -2.0)
axs[1].set_xlabel("c (eje invertido)"); axs[1].set_ylabel("$z_n$")
axs[1].set_title("$z_{n+1}=z_n^2+c$ sobre el eje real (−2 ≤ c ≤ 0.25)")
for c_ in RES["reto"]["c_bif"][:4]:
    axs[1].axvline(c_, color="tab:red", lw=0.6, ls="--")
fig.tight_layout()
fig.savefig(f"{OUT}/reto.png", dpi=115)
plt.close(fig)

# Figura: Mandelbrot con eje real marcado
N, vivo = mandelbrot(-2.1, 0.6, -1.3, 1.3, 800, 770, 200)
fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap("twilight_shifted").copy(); cmap.set_bad("black")
ax.imshow(N, extent=(-2.1, 0.6, -1.3, 1.3), origin="lower", cmap=cmap, vmin=1, vmax=200)
for c_, lab in zip(RES["reto"]["c_bif"][:4], ["p=1→2", "2→4", "4→8", "8→16"]):
    ax.plot([c_], [0], "o", color="white", ms=4)
ax.plot([RES["reto"]["c_inf"]], [0], "*", color="yellow", ms=10, label="$c_\\infty$ (Feigenbaum)")
ax.axhline(0, color="white", lw=0.4)
ax.set_xlabel("Re(c)"); ax.set_ylabel("Im(c)")
ax.set_title("Conjunto de Mandelbrot: puntos de duplicación sobre el eje real")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/reto_mandel.png", dpi=110)
plt.close(fig)

with open(os.path.join(BASE_DIR, "resultados.json"), "w") as f:
    json.dump(RES, f, indent=1, default=str)
print("OK")
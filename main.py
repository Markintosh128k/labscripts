import numpy as np  # noqa: I001
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.odr import ODR, Model, RealData


def getData(filepath):
    df = pd.read_excel(filepath)
    x, y, dx, dy = df.to_numpy(dtype=float).T
    labels = list(df.columns)
    return x, y, dx, dy, labels


def linEsponenziale(a, tau, x):
    return retta((-(1 / tau), np.log(a)), x)


def linPotenza(a, x, b):
    return retta((b, np.log(a)), np.log(x))


def chiQuadroLinare(x, y, m, q, dx, dy):
    residui = (y - m * x - q) ** 2
    errori = dy**2 + m**2 * dx**2
    return np.sum(residui / errori)


def retta(params, x):
    m, q = params
    return m * x + q


def pulisci(nome):
    return nome.split("=")[-1].strip()


def parametriFitODR(x, y, dx, dy, stima_iniziale=(1.0, 0.0)):
    modello = Model(retta)
    dati = RealData(x, y, sx=dx, sy=dy)
    odr = ODR(dati, modello, beta0=stima_iniziale)
    risultato = odr.run()

    m, q = risultato.beta
    dm, dq = np.sqrt(np.diag(risultato.cov_beta))

    return m, q, dm, dq


def consistenzaStatistica(a, b, da, db, k):
    discrepanza = abs(a - b)
    sigma_d = np.sqrt(da**2 + db**2)

    return discrepanza <= k * sigma_d


if __name__ == "__main__":
    # Configurazione a monte
    xm, ym, dxm, dym, nomi = getData(sys.argv[1])
    # Configurazione a valle
    xv, yv, dxv, dyv, nomi = getData(sys.argv[2])
    k = 2

    nomi = [pulisci(n) for n in nomi]
    print(f"Caricati {len(xm)} punti da {sys.argv[1]}")
    print(f"Caricati {len(xv)} punti da {sys.argv[2]}")

    rA, rV = 0.588, 40.0
    mm, qm, dmm, dqm = parametriFitODR(xm, ym, dxm, dym)
    mv, qv, dmv, dqv = parametriFitODR(xv, yv, dxv, dyv)

    Rx_v, dRx_v = mv - rA, dmv
    Rx_m, dRx_m = (mm * rV) / (rV - mm), rV**2 / (rV - mm) ** 2 * dmm
    t = abs(Rx_v - Rx_m) / np.hypot(dRx_v, dRx_m)
    w = np.array([1 / dRx_v**2, 1 / dRx_m**2])
    Rx = (w[0] * Rx_v + w[1] * Rx_m) / w.sum()
    dRx = 1 / np.sqrt(w.sum())

    z = abs(Rx_v - Rx_m) / np.sqrt(dRx_v**2 + dRx_m**2)

    print("=========================================")
    print("Configurazione a monte:")
    print(f"A' = {mm:.3f} \u00b1 {dmm:.3f} k\u03a9")
    print(f"B' = {qm:.4f} \u00b1 {dqm:.4f} V")
    print(f"Rx = {Rx_m:.3f} \u00b1 {dRx_m:.3f} k\u03a9")
    print("=========================================")
    print("Configurazione a valle:")
    print(f"A  = {mv:.3f} \u00b1 {dmv:.3f} k\u03a9")
    print(f"B  = {qv:.4f} \u00b1 {dqv:.4f} V")
    print(f"Rx = {Rx_v:.3f} \u00b1 {dRx_v:.3f} k\u03a9")
    print("=========================================")
    print(f"Discrepanza: {z:.2f} sigma")
    if consistenzaStatistica(Rx_v, Rx_m, dRx_v, dRx_m, k):
        print(f"Le stime risultano compatibili entro {k} sigma")
        print(f"Media pesata: Rx = {Rx:.3f} \u00b1 {dRx:.3f} k\u03a9")
    else:
        print(
            f"Le stime NON risultano compatibili entro {k} sigma: media pesata non significativa"
        )
    print("=========================================")
    chi2m = chiQuadroLinare(xm, ym, mm, qm, dxm, dym)
    gdlm = xm.size - 2
    print(f"chi2 monte: {chi2m:.3f}  chi2/gdl: {chi2m / gdlm:.3f}")

    chi2v = chiQuadroLinare(xv, yv, mv, qv, dxv, dyv)
    gdlv = xv.size - 2
    print(f"chi2 valle: {chi2v:.3f}  chi2/gdl: {chi2v / gdlv:.3f}")

    # Curve di fit (calcolate una volta, riusate nei tre grafici)
    x_fit_m = np.linspace(xm.min(), xm.max(), 200)
    y_fit_m = retta((mm, qm), x_fit_m)
    x_fit_v = np.linspace(xv.min(), xv.max(), 200)
    y_fit_v = retta((mv, qv), x_fit_v)

    # ===== Grafico 1: monte e valle insieme =====
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        xm,
        ym,
        xerr=dxm,
        yerr=dym,
        fmt="o",
        color="blue",
        ecolor="blue",
        capsize=2,
        elinewidth=0.8,
        markersize=2,
        alpha=0.85,
        label="Dati (conf. a monte)",
    )
    ax.plot(x_fit_m, y_fit_m, "-", color="blue", linewidth=1.5)

    ax.errorbar(
        xv,
        yv,
        xerr=dxv,
        yerr=dyv,
        fmt="o",
        color="red",
        ecolor="red",
        capsize=2,
        elinewidth=0.8,
        markersize=2,
        alpha=0.85,
        label="Dati (conf. a valle)",
    )
    ax.plot(x_fit_v, y_fit_v, "-", color="red", linewidth=1.5)

    ax.set_xlabel(nomi[0])
    ax.set_ylabel(nomi[1])
    ax.set_title("Stima di R col metodo voltamperometrico")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ===== Grafico 2: solo configurazione a monte =====
    fig_m, ax_m = plt.subplots(figsize=(8, 6))

    ax_m.errorbar(
        xm,
        ym,
        xerr=dxm,
        yerr=dym,
        fmt="o",
        color="blue",
        ecolor="blue",
        capsize=2,
        elinewidth=0.8,
        markersize=2,
        alpha=0.85,
        label="Dati (conf. a monte)",
    )
    ax_m.plot(
        x_fit_m,
        y_fit_m,
        "-",
        color="blue",
        linewidth=1.5,
        label=f"Fit: A' = {mm:.3f} \u00b1 {dmm:.3f} k\u03a9",
    )

    ax_m.set_xlabel(nomi[0])
    ax_m.set_ylabel(nomi[1])
    ax_m.set_title("Configurazione a monte")
    ax_m.grid(True, alpha=0.3)
    ax_m.legend()

    # ===== Grafico 3: solo configurazione a valle =====
    fig_v, ax_v = plt.subplots(figsize=(8, 6))

    ax_v.errorbar(
        xv,
        yv,
        xerr=dxv,
        yerr=dyv,
        fmt="o",
        color="red",
        ecolor="red",
        capsize=2,
        elinewidth=0.8,
        markersize=2,
        alpha=0.85,
        label="Dati (conf. a valle)",
    )
    ax_v.plot(
        x_fit_v,
        y_fit_v,
        "-",
        color="red",
        linewidth=1.5,
        label=f"Fit: A = {mv:.3f} \u00b1 {dmv:.3f} k\u03a9",
    )

    ax_v.set_xlabel(nomi[0])
    ax_v.set_ylabel(nomi[1])
    ax_v.set_title("Configurazione a valle")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    plt.show()

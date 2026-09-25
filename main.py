import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.odr import ODR, Model, RealData
from scipy.stats import norm


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


def residui(x, y, dx, dy, m, q):
    r = y - retta((m, q), x)
    sr = np.sqrt(dy**2 + m**2 * dx**2)
    return r, sr


def consistenzaStatistica(a, b, da, db, k):
    discrepanza = abs(a - b)
    sigma_d = np.sqrt(da**2 + db**2)

    return discrepanza <= k * sigma_d


if __name__ == "__main__":
    # Configurazione a monte
    xm, ym, dxm, dym, nomi = getData(sys.argv[1])
    # Configurazione a valle
    xv, yv, dxv, dyv, nomi = getData(sys.argv[2])
    k = 3

    nomi = [pulisci(n) for n in nomi]
    print(f"Caricati {len(xm)} punti da {sys.argv[1]}")
    print(f"Caricati {len(xv)} punti da {sys.argv[2]}")

    rA, rV = 0.558, 40.0  # kΩ (valori forniti dal professore)
    mm, qm, dmm, dqm = parametriFitODR(xm, ym, dxm, dym)
    mv, qv, dmv, dqv = parametriFitODR(xv, yv, dxv, dyv)

    Rx_v, dRx_v = mv - rA, dmv
    Rx_m, dRx_m = (mm * rV) / (rV - mm), rV**2 / (rV - mm) ** 2 * dmm
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

    rm, srm = residui(xm, ym, dxm, dym, mm, qm)
    rv, srv = residui(xv, yv, dxv, dyv, mv, qv)

    # Curve di fit
    x_fit_m = np.linspace(0, xm.max(), 200)
    y_fit_m = retta((mm, qm), x_fit_m)
    x_fit_v = np.linspace(0, xv.max(), 200)
    y_fit_v = retta((mv, qv), x_fit_v)

    stile = dict(capsize=2, elinewidth=0.8, markersize=5, alpha=0.85)

    # ===== Finestra 1: dati e rette di fit =====
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        xm,
        ym,
        xerr=dxm,
        yerr=dym,
        fmt="o",
        color="blue",
        label="Dati (conf. a monte)",
        **stile,
    )
    ax.plot(
        x_fit_m,
        y_fit_m,
        "-",
        color="blue",
        linewidth=1.5,
        label=f"Fit monte: A' = {mm:.3f} \u00b1 {dmm:.3f} k\u03a9",
    )

    ax.errorbar(
        xv,
        yv,
        xerr=dxv,
        yerr=dyv,
        fmt="s",
        color="red",
        markerfacecolor="none",
        label="Dati (conf. a valle)",
        **stile,
    )
    ax.plot(
        x_fit_v,
        y_fit_v,
        "--",
        color="red",
        linewidth=1.5,
        label=f"Fit valle: A = {mv:.3f} \u00b1 {dmv:.3f} k\u03a9",
    )

    ax.set_xlabel(nomi[0])
    ax.set_ylabel(nomi[1])
    ax.set_title("Stima di R col metodo voltamperometrico")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()

    # ===== Finestra 2: residui normalizzati + istogramma a lato =====
    zm, zv = rm / srm, rv / srv
    edges = np.linspace(-3, 3, 7)  # 6 bin larghi 1 sigma
    y_g = np.linspace(-3.5, 3.5, 400)

    fig_r, axs_r = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        sharex="col",
        sharey=True,
        width_ratios=[4, 1],
        layout="constrained",
    )
    dati = [
        (xm, zm, "o", "blue", "monte", chi2m / gdlm),
        (xv, zv, "s", "red", "valle", chi2v / gdlv),
    ]
    for (ax_s, ax_h), (xi, zi, mk, col, nome, rchi2) in zip(axs_r, dati):
        # sinistra: residui normalizzati in funzione di I, fasce a 1 e 2 sigma
        ax_s.axhspan(-1, 1, color="gray", alpha=0.15, lw=0)
        ax_s.axhspan(-2, 2, color="gray", alpha=0.08, lw=0)
        ax_s.axhline(0, color="black", linewidth=0.8)
        ax_s.plot(
            xi,
            zi,
            mk,
            color=col,
            markerfacecolor="none" if mk == "s" else col,
            label=f"Residui {nome} ($\\chi^2$/gdl = {rchi2:.2f})",
        )
        ax_s.set_ylabel(r"$r_i / \sigma_{r_i}$")
        ax_s.set_ylim(-3.5, 3.5)
        ax_s.grid(True, alpha=0.3)
        ax_s.legend(loc="upper left")

        # destra: istogramma orizzontale + gaussiana N(0,1) attesa
        ax_h.hist(
            zi,
            bins=edges,
            orientation="horizontal",
            color=col,
            alpha=0.5,
            edgecolor="black",
        )
        ax_h.plot(
            len(zi) * (edges[1] - edges[0]) * norm.pdf(y_g), y_g, "k-", linewidth=1
        )
        ax_h.grid(True, alpha=0.3)

    axs_r[0, 0].set_title("Residui normalizzati dei fit lineari")
    axs_r[1, 0].set_xlabel(nomi[0])
    axs_r[1, 1].set_xlabel("conteggi")

    plt.show()


from pathlib import Path

import matplotlib.pyplot as plt
import Manage_file as mf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==================================================
# CARTELLA DI SALVATAGGIO
# ==================================================

IMG_FOLDER = Path(__file__).parent / "img"
IMG_FOLDER.mkdir(parents=True, exist_ok=True)


# ==================================================
# FUNZIONE PER SALVARE I GRAFICI
# ==================================================

def save_plot(filename):
    """
    Salva il grafico nella cartella img del progetto
    e chiude la figura.
    """
    output_path = IMG_FOLDER / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grafico salvato: {output_path}")


# ==================================================
# DISTRIBUZIONE DEL NUMERO DI CLICK
# ==================================================

def plot_distributions(counts, filename="distribuzione.png"):
    """
    Confronta la distribuzione sperimentale con le distribuzioni
    teoriche di Poisson e Bose-Einstein.
    """
    counts = np.asarray(counts, dtype=int)

    if counts.size == 0:
        raise ValueError("Array dei conteggi vuoto.")

    p_exp = mf.number_of_clicks(counts)
    p_poisson = mf.poisson_distribution(counts)
    p_be = mf.bose_einstain_distribution(counts)

    n = np.arange(len(p_exp))

    plt.figure(figsize=(8, 5))
    plt.plot(n, p_exp, "o", label="Experimental")
    plt.plot(n, p_poisson, "-", label="Poisson")
    plt.plot(n, p_be, "-", label="Bose-Einstein")

    plt.xlabel("Number of clicks n")
    plt.ylabel("Probability p(n)")
    plt.title("Photon Number Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot(filename)


# ==================================================
# FATTORE DI FANO
# ==================================================

def plot_fano_vs_bin_duration(
    time_tags_seconds,
    bin_durations,
    filename="fano.png"
):
    """
    Mostra il fattore di Fano in funzione della durata del bin.
    """
    durations, fano_values = mf.fano_vs_bin_duration(
        time_tags_seconds,
        bin_durations
    )

    plt.figure(figsize=(8, 5))
    plt.plot(durations, fano_values, "o-")

    plt.xscale("log")
    plt.xlabel("Bin duration (s)")
    plt.ylabel("Fano factor F")
    plt.title("Fano Factor vs Bin Duration")
    plt.grid(True, which="both", alpha=0.3)

    save_plot(filename)

    return durations, fano_values


# ==================================================
# ISTOGRAMMA DEGLI INTERVALLI TRA CLICK
# ==================================================

def plot_interarrival_histogram(
    time_tags_seconds,
    bins=100,
    filename="intervalli.png"
):
    """
    Mostra l'istogramma degli intervalli tra click consecutivi.
    Evidenzia la soglia di 200 ns per gli afterpulse.
    """
    intervals = mf.interarrival_times(time_tags_seconds)

    if intervals.size == 0:
        raise ValueError(
            "Non ci sono intervalli temporali da mostrare."
        )

    intervals = intervals[intervals > 0]

    if intervals.size == 0:
        raise ValueError(
            "Non ci sono intervalli temporali positivi."
        )

    plt.figure(figsize=(8, 5))
    hist, edges, _ = plt.hist(
        intervals,
        bins=bins,
        log=True,
        alpha=0.7,
        label="Data"
    )

    A, tau = mf.fit_interarrival_distribution(
        time_tags_seconds
    )

    x = np.linspace(
        intervals.min(),
        intervals.max(),
        1000
    )

    plt.plot(
        x,
        mf.exponential(x, A, tau),
        "r-",
        linewidth=2,
        label=f"Fit tau={tau:.2e} s"
    )
    plt.axvline(
        200e-9,
        color="red",
        linestyle="--",
        label="Afterpulse threshold: 200 ns"
    )

    plt.xlabel("Time interval between consecutive clicks (s)")
    plt.ylabel("Number of intervals (log scale)")
    plt.title("Interarrival Time Distribution")

    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot(filename)


# ==================================================
# CONFRONTO PRIMA E DOPO IL FILTRO AFTERPULSE
# ==================================================

def plot_before_after_filter(
    counts_before,
    counts_after,
    filename="filtro_afterpulse.png"
):
    """
    Confronta le distribuzioni sperimentali prima e dopo
    il filtro degli afterpulse.
    """
    p_before = mf.number_of_clicks(counts_before)
    p_after = mf.number_of_clicks(counts_after)

    n_before = np.arange(len(p_before))
    n_after = np.arange(len(p_after))

    plt.figure(figsize=(8, 5))
    plt.plot(
        n_before,
        p_before,
        "o-",
        label="Before filtering"
    )
    plt.plot(
        n_after,
        p_after,
        "s-",
        label="After filtering"
    )

    plt.xlabel("Number of clicks n")
    plt.ylabel("Probability p(n)")
    plt.title("Photon Number Distribution Before and After Afterpulse Removal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot(filename)

def plot_byte_histogram(
    byte_stream,
    filename="byte_histogram.png"
):

    plt.figure(figsize=(10,5))

    plt.hist(
        byte_stream,
        bins=256
    )

    mean_count = (
        len(byte_stream)
        / 256
    )

    plt.axhline(
        mean_count,
        color="red",
        label="Mean"
    )

    plt.xlabel("Byte value")
    plt.ylabel("Occurrences")

    plt.title(
        "Distribution of Generated Random Bytes"
    )

    plt.legend()
    plt.grid(True)

    save_plot(filename)
def plot_byte_cloud(
    byte_stream,
    filename="byte_cloud.png"
):

    print(
        f"Bytes available for cloud: "
        f"{len(byte_stream)}"
    )

    if len(byte_stream) < 3:
        print(
            "Not enough bytes for 3D cloud."
        )
        return

    fig = plt.figure(
        figsize=(8, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    x = byte_stream[:-2]
    y = byte_stream[1:-1]
    z = byte_stream[2:]

    ax.scatter(
        x,
        y,
        z,
        s=1
    )

    ax.set_xlabel("Byte i")
    ax.set_ylabel("Byte i+1")
    ax.set_zlabel("Byte i+2")

    output_path = IMG_FOLDER / filename

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

def plot_bit_balance(
    zeros,
    ones,
    filename="bit_balance.png"
):

    plt.figure(figsize=(6,4))

    plt.bar(
        ["0", "1"],
        [zeros, ones]
    )

    plt.xlabel("Bit")
    plt.ylabel("Occurrences")

    plt.title(
        "Bit Frequency Test"
    )

    plt.grid(True)

    save_plot(filename)

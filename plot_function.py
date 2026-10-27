from pathlib import Path

import matplotlib.pyplot as plt
import Manage_file as mf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==================================================
# OUTPUT FOLDER
# ==================================================

IMG_FOLDER = Path(__file__).parent / "img"
IMG_FOLDER.mkdir(parents=True, exist_ok=True)


# ==================================================
# FUNCTION TO SAVE PLOTS
# ==================================================

def save_plot(filename):
    """
    Saves the plot in the project's img folder
    and closes the figure.
    """
    output_path = IMG_FOLDER / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved: {output_path}")


# ==================================================
# CLICK NUMBER DISTRIBUTION
# ==================================================

def plot_distributions(counts, filename="distribuzione.png"):
    """
    Compares the experimental distribution with the theoretical
    Poisson and Bose-Einstein distributions.
    """
    counts = np.asarray(counts, dtype=int)

    if counts.size == 0:
        raise ValueError("The counts array is empty.")

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
# COMPARISON OF p(n) FOR DIFFERENT BIN DURATIONS
# ==================================================

def plot_distributions_different_bins(
    time_tags_seconds,
    bin_durations,
    filename="different_bins.png"
):
    """
    Compares the probability distributions p(n) obtained
    using different bin durations.
    """

    plt.figure(figsize=(8, 5))

    for dt in bin_durations:

        counts = mf.count_interval(
            time_tags_seconds,
            delta_interval=dt
        )

        p_n = mf.number_of_clicks(counts)
        n = np.arange(len(p_n))

        plt.plot(
            n,
            p_n,
            marker="o",
            linestyle="-",
            label=f"{dt*1e6:.0f} us"
        )

    plt.xlabel("Number of clicks n")
    plt.ylabel("Probability p(n)")
    plt.title("Photon Number Distribution for Different Bin Durations")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot(filename)


# ==================================================
# FANO FACTOR
# ==================================================

def plot_fano_vs_bin_duration(
    time_tags_seconds,
    bin_durations,
    background_mean=None,
    filename="fano.png"
):
    """
    Plots the Fano factor as a function of bin duration.
    """
    durations, fano_values = mf.fano_vs_bin_duration(
        time_tags_seconds,
        bin_durations,
        background_mean=background_mean
    )

    plt.figure(figsize=(8, 5))
    plt.plot(durations, fano_values, "o-")

    plt.axhline(
        1,
        color="red",
        linestyle="--",
        label="Poisson, F = 1"
    )

    plt.xscale("log")
    plt.xlabel("Bin duration (s)")
    plt.ylabel("Fano factor F")
    plt.title("Fano Factor vs Bin Duration")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    save_plot(filename)

    return durations, fano_values


# ==================================================
# INTERARRIVAL TIME HISTOGRAM
# ==================================================

def plot_interarrival_histogram(
    time_tags_seconds,
    bins=100,
    threshold=200e-9,
    filename="intervalli.png"
):
    """
    Plots a histogram of the time intervals between consecutive clicks,
    with a logarithmic y-axis and an exponential fit.
    """
    intervals = mf.interarrival_times(time_tags_seconds)
    intervals = intervals[intervals > 0]

    if intervals.size == 0:
        raise ValueError("There are no time intervals to display.")

    plt.figure(figsize=(8, 5))

    plt.hist(
        intervals,
        bins=bins,
        density=True,
        log=True,
        alpha=0.7,
        label="Data"
    )

    tau = mf.exponential_tau(
        time_tags_seconds,
        threshold=threshold
    )

    x = np.linspace(threshold, intervals.max(), 1000)
    y = (1 / tau) * np.exp(-x / tau)

    plt.plot(
        x,
        y,
        "r-",
        linewidth=2,
        label=rf"Exponential fit, $\tau$ = {tau:.2e} s"
    )

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        label="Afterpulse threshold"
    )

    plt.xscale("log")
    plt.xlabel("Time interval between consecutive clicks (s)")
    plt.ylabel("Probability density")
    plt.title("Interarrival Time Distribution")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    save_plot(filename)

    return tau


# ==================================================
# COMPARISON BEFORE AND AFTER AFTERPULSE FILTERING
# ==================================================

def plot_before_after_filter(
    counts_before,
    counts_after,
    filename="filtro_afterpulse.png"
):
    """
    Compares the experimental distributions before and after
    afterpulse filtering.
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

    plt.figure(figsize=(10, 5))

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

    plt.figure(figsize=(6, 4))

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


def plot_count_autocorrelation(counts, max_lag=20, filename="autocorrelation.png"):
    ac = mf.count_autocorrelation(counts, max_lag=max_lag)
    lags = np.arange(1, max_lag + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(lags, ac, "o-")
    plt.axhline(0, color="black", linestyle="--")

    plt.xlabel("Lag (bin)")
    plt.ylabel("Normalized autocorrelation")
    plt.title("Count Autocorrelation")
    plt.grid(True, alpha=0.3)

    save_plot(filename)


def plot_byte_comb_check(byte_stream, filename="byte_comb_check.png"):
    """
    Looks for the byte comb pattern, as shown in slides 27–29:
    a 256-bin histogram with mean, standard deviation, and sqrt(mean) lines.
    """
    plt.figure(figsize=(10, 5))
    hcomb, xedges = np.histogram(byte_stream, bins=256, range=(0, 256))
    bin_centers = (xedges[:-1] + xedges[1:]) / 2

    plt.bar(bin_centers, hcomb, width=1.0, alpha=0.7, label="Byte counts")

    hmean = np.mean(hcomb)
    stdhcomb = np.std(hcomb)

    plt.axhline(hmean, color="red", linestyle="-", label=f"Mean = {hmean:.1f}")
    plt.axhline(hmean - stdhcomb, color="red", linestyle="--", label="± Std Dev")
    plt.axhline(hmean + stdhcomb, color="red", linestyle="--")
    plt.axhline(hmean - np.sqrt(hmean), color="black", linestyle=":", label="± sqrt(Mean)")
    plt.axhline(hmean + np.sqrt(hmean), color="black", linestyle=":")

    plt.xlabel("Byte value (0-255)")
    plt.ylabel("Occurrences")
    plt.title("After the Extraction: Check for the Comb")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)

    save_plot(filename)


def plot_comb_distribution(byte_stream, filename="comb_distribution.png"):
    """
    Plots a secondary histogram of the distribution of byte frequencies
    (slides 27 and 29).
    """
    hcomb, _ = np.histogram(byte_stream, bins=256, range=(0, 256))

    plt.figure(figsize=(8, 5))
    plt.hist(hcomb, bins=20, alpha=0.8, edgecolor="black")
    plt.xlabel("Number of byte appearances")
    plt.ylabel("Frequency")
    plt.title("Comb Distribution around the Average Number of Byte Appearances")
    plt.grid(True, alpha=0.3)

    save_plot(filename)
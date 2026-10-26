from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import traceback
import numpy as np

import Manage_file as mf
import plot_function as pf


# ==================================================
# PERCORSI
# ==================================================

BASE_FOLDER = Path(__file__).parent
DATA_FOLDER = BASE_FOLDER / "data_files"
IMG_FOLDER = BASE_FOLDER / "img"
LOG_PATH = BASE_FOLDER / "log.txt"

IMG_FOLDER.mkdir(parents=True, exist_ok=True)


# ==================================================
# PARAMETRI DELL'ANALISI
# ==================================================

BIN_DURATION = 1e-5  # 10 microsecondi
AFTERPULSE_THRESHOLD = 200e-9  # 200 ns

BIN_DURATIONS = np.array([
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
])


# File delle misure alle diverse tensioni
MEASUREMENT_FILES = {
    "0V": "TimeTags_0v_15.txt",
    "1V": "TimeTags_1v_15.txt",
    "3V": "TimeTags_3v_15.txt",
    "5V": "TimeTags_5v_15.txt",
}


# File di background
BACKGROUND_FILES = {
    "laser_off": "TimeTags_background_laser_off.txt",
    "no_fiber": "TimeTags_background_no_fiber.txt",
}


# ==================================================
# ANALISI DELLE MISURE
# ==================================================

def analyze_measurement(voltage, filename, background_mean):
    print("\n" + "=" * 60)
    print(f"ANALISI MISURA: {voltage}")
    print(f"File: {filename}")
    print("=" * 60)

    file_path = DATA_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}"
        )

    # Lettura dei time tag e conversione in secondi
    time_tags = mf.reader(file_path)
    times = mf.converter(time_tags)

    if len(times) == 0:
        print("Il file non contiene time tag.")
        return

    print(f"Numero totale di click: {len(times)}")

    # ----------------------------------------------
    # Conteggi nei bin temporali
    # ----------------------------------------------

    counts = mf.count_interval(
        times,
        delta_interval=BIN_DURATION
    )

    print(f"Durata del bin: {BIN_DURATION:.3e} s")
    print(f"Numero di bin conteggiati: {len(counts)}")

    # ----------------------------------------------
    # Autocorrelazione
    # ----------------------------------------------

    pf.plot_count_autocorrelation(
        counts,
        max_lag=20,
        filename=f"autocorrelazione_{voltage}.png"
    )

    # ----------------------------------------------
    # Media e momenti centrali
    # ----------------------------------------------

    mean, mu2, mu3, mu4 = mf.avg_and_central_moments(
        counts
    )

    print(f"Media dei click per bin: {mean:.8g}")
    print(f"Momento centrale ordine 2: {mu2:.8g}")
    print(f"Momento centrale ordine 3: {mu3:.8g}")
    print(f"Momento centrale ordine 4: {mu4:.8g}")

    # ----------------------------------------------
    # Fattore di Fano prima del filtro
    # ----------------------------------------------

    fano = mf.fano_factor(
        counts,
        background_mean=background_mean
    )

    print(f"Fattore di Fano corretto per background: {fano:.8g}")

    # ----------------------------------------------
    # Grafico della distribuzione dei click
    # ----------------------------------------------

    pf.plot_distributions(
        counts,
        filename=f"distribuzione_{voltage}.png"
    )

    pf.plot_distributions_different_bins(
        times,
        [1e-6, 1e-5, 5e-5],
        filename=f"distribuzioni_bin_{voltage}.png"
    )
    # ----------------------------------------------
    # Fano factor prima del filtro
    # ----------------------------------------------

    pf.plot_fano_vs_bin_duration(
        times,
        BIN_DURATIONS,
        background_mean=background_mean,
        filename=f"fano_{voltage}.png"
    )

    # ----------------------------------------------
    # Istogramma degli intervalli prima del filtro
    # ----------------------------------------------

    pf.plot_interarrival_histogram(
        times,
        threshold=AFTERPULSE_THRESHOLD,
        filename=f"intervalli_{voltage}.png"
    )

    # ----------------------------------------------
    # Tau della distribuzione esponenziale
    # ----------------------------------------------

    tau = mf.exponential_tau(
        times,
        threshold=AFTERPULSE_THRESHOLD
    )

    print(f"Exponential decay time tau = {tau:.6e} s")

    # ----------------------------------------------
    # Filtro degli afterpulse
    # ----------------------------------------------

    filtered_times = mf.remove_afterpulses(
        times,
        threshold=AFTERPULSE_THRESHOLD
    )

    removed_clicks = len(times) - len(filtered_times)

    print(f"Click dopo il filtro: {len(filtered_times)}")
    print(f"Click rimossi dal filtro: {removed_clicks}")

    if len(times) > 0:
        percentage_removed = 100 * removed_clicks / len(times)
        print(f"Percentuale di click rimossi: {percentage_removed:.4f}%")

    # ----------------------------------------------
    # Conteggi dopo il filtro
    # ----------------------------------------------

    counts_after = mf.count_interval(
        filtered_times,
        delta_interval=BIN_DURATION
    )

    # ----------------------------------------------
    # Fano factor dopo il filtro
    # ----------------------------------------------

    pf.plot_fano_vs_bin_duration(
        filtered_times,
        BIN_DURATIONS,
        background_mean=background_mean,
        filename=f"fano_filtered_{voltage}.png"
    )

    # ----------------------------------------------
    # Istogramma degli intervalli dopo il filtro
    # ----------------------------------------------

    pf.plot_interarrival_histogram(
        filtered_times,
        threshold=AFTERPULSE_THRESHOLD,
        filename=f"intervalli_filtered_{voltage}.png"
    )

    # ----------------------------------------------
    # Media e momenti dopo il filtro
    # ----------------------------------------------

    mean_after, mu2_after, mu3_after, mu4_after = \
        mf.avg_and_central_moments(
            counts_after
        )

    fano_after = mf.fano_factor(
        counts_after,
        background_mean=background_mean
    )

    print("\nAFTERPULSE FILTERED DATA")

    print(f"Mean = {mean_after:.8g}")
    print(f"2nd central moment = {mu2_after:.8g}")
    print(f"3rd central moment = {mu3_after:.8g}")
    print(f"4th central moment = {mu4_after:.8g}")
    print(f"Fano factor = {fano_after:.8g}")

    # ----------------------------------------------
    # Confronto prima e dopo il filtro
    # ----------------------------------------------

    pf.plot_before_after_filter(
        counts,
        counts_after,
        filename=f"filtro_afterpulse_{voltage}.png"
    )

    print(f"Grafici salvati nella cartella: {IMG_FOLDER}")

    # ----------------------------------------------
    # Momenti teorici
    # ----------------------------------------------

    poisson_mu2, poisson_mu3, poisson_mu4 = \
        mf.poisson_moments(mean)

    be_mu2, be_mu3, be_mu4 = \
        mf.bose_einstein_moments(mean)

    print("\nTheoretical moments:")

    print("Poisson:")
    print(f"mu2 = {poisson_mu2:.8g}")
    print(f"mu3 = {poisson_mu3:.8g}")
    print(f"mu4 = {poisson_mu4:.8g}")

    print("\nBose-Einstein:")
    print(f"mu2 = {be_mu2:.8g}")
    print(f"mu3 = {be_mu3:.8g}")
    print(f"mu4 = {be_mu4:.8g}")

    skewness, kurtosis = \
        mf.skewness_and_kurtosis(counts)

    print(f"Skewness = {skewness:.8g}")
    print(f"Kurtosis = {kurtosis:.8g}")

    # ----------------------------------------------
    # Random number generator
    # ----------------------------------------------

    print("\nRANDOM NUMBER GENERATOR")

    bits = mf.generate_random_bits(
        filtered_times,
        threshold=AFTERPULSE_THRESHOLD
    )

    print(f"Generated bits: {len(bits)}")

    bytes_stream = mf.bits_to_bytes(bits)

    print(f"Generated bytes: {len(bytes_stream)}")

    chi2, p_value = mf.chi_square_test(bytes_stream)

    print(f"Chi-square = {chi2:.4f}")
    print(f"p-value = {p_value:.4f}")

    mean_byte = mf.byte_mean(bytes_stream)

    print(f"Byte mean = {mean_byte:.4f}")

    serial_corr = mf.serial_correlation(bytes_stream)

    print(f"Serial correlation = {serial_corr:.6f}")

    pf.plot_byte_histogram(
        bytes_stream,
        filename=f"random_bytes_{voltage}.png"
    )

    zeros, ones = mf.bit_balance(bits)

    print(f"Zeros = {zeros}")
    print(f"Ones = {ones}")

    if len(bits) > 0:
        print(f"Fraction of ones = {ones / len(bits):.4f}")

    pi_estimate = mf.monte_carlo_pi(bytes_stream)

    print(f"Monte Carlo Pi = {pi_estimate:.6f}")

    pf.plot_byte_cloud(
        bytes_stream,
        filename=f"byte_cloud_{voltage}.png"
    )

    pf.plot_bit_balance(
        zeros,
        ones,
        filename=f"bit_balance_{voltage}.png"
    )


# ==================================================
# ANALISI DEL BACKGROUND
# ==================================================

def analyze_background(background_name, filename):
    print("\n" + "=" * 60)
    print(f"ANALISI BACKGROUND: {background_name}")
    print(f"File: {filename}")
    print("=" * 60)

    file_path = DATA_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}"
        )

    background_tags = mf.reader(file_path)
    background_times = mf.converter(background_tags)

    filtered_background = mf.remove_afterpulses(
        background_times,
        threshold=AFTERPULSE_THRESHOLD
    )

    removed = len(background_times) - len(filtered_background)

    print(f"Removed background clicks: {removed}")

    if len(filtered_background) < 2:
        print("Dati insufficienti per stimare la durata.")
        return

    # Stima della durata tra primo e ultimo time tag
    background_duration = (
        filtered_background[-1]
        - filtered_background[0]
    )

    if background_duration <= 0:
        print("Durata non valida: impossibile calcolare il rate.")
        return

    # Rate di background in click/s
    background_rate = mf.background_rate(
        filtered_background,
        duration=background_duration
    )

    # Numero medio atteso di click di background per bin
    mean_background = mf.background_mean(
        background_rate,
        delta_interval=BIN_DURATION
    )

    print(f"Numero di click: {len(background_times)}")
    print(f"Durata stimata: {background_duration:.8g} s")
    print(f"Rate di background: {background_rate:.8g} click/s")
    print(f"Click medi di background per bin: {mean_background:.8g}")


# ==================================================
# FUNZIONE PRINCIPALE
# ==================================================

def main():
    print("ANALISI STATISTICA DEI PHOTON TIME TAGS")
    print(f"Cartella dati: {DATA_FOLDER}")
    print(f"Cartella immagini: {IMG_FOLDER}")
    print(f"Durata bin principale: {BIN_DURATION:.3e} s")
    print(f"Soglia afterpulse: {AFTERPULSE_THRESHOLD * 1e9:.0f} ns")

    # Calcolo del background medio per bin
    background_rates = {}
    background_means = {}

    for background_name, background_filename in BACKGROUND_FILES.items():
        background_path = DATA_FOLDER / background_filename
        background_tags = mf.reader(background_path)
        background_times = mf.converter(background_tags)

        background_times = mf.remove_afterpulses(
            background_times,
            threshold=AFTERPULSE_THRESHOLD
        )

        rate = mf.background_rate(background_times)
        mean_bg = mf.background_mean(
            rate,
            delta_interval=BIN_DURATION
        )

        background_rates[background_name] = rate
        background_means[background_name] = mean_bg

        print(f"\nBackground: {background_name}")
        print(f"Rate = {rate:.8g} click/s")
        print(f"Mean background per bin = {mean_bg:.8g} click")

    # Usa la media dei due background disponibili
    BACKGROUND_MEAN = np.mean(list(background_means.values()))

    print(
        f"\nMean background used for correction = "
        f"{BACKGROUND_MEAN:.8g}"
    )

    # Analisi delle quattro tensioni
    for voltage, filename in MEASUREMENT_FILES.items():
        try:
            analyze_measurement(
                voltage,
                filename,
                background_mean=BACKGROUND_MEAN
            )
        except Exception:
            print(f"\nERRORE durante l'analisi di {voltage}:")
            traceback.print_exc()

    # Analisi dei due file di background
    for background_name, filename in BACKGROUND_FILES.items():
        try:
            analyze_background(background_name, filename)
        except Exception:
            print(
                f"\nERRORE durante l'analisi del background "
                f"{background_name}:"
            )
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ANALISI COMPLETATA")
    print(f"Immagini salvate in: {IMG_FOLDER}")
    print("=" * 60)


# ==================================================
# AVVIO E SALVATAGGIO DEL LOG
# ==================================================

if __name__ == "__main__":
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            try:
                main()
            except Exception:
                print("ERRORE GENERALE:")
                traceback.print_exc()

    print("Esecuzione terminata.")
    print(f"Log salvato in: {LOG_PATH}")
    print(f"Grafici salvati in: {IMG_FOLDER}")




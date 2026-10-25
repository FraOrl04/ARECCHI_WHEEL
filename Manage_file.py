import pandas as pd
import numpy as np
from scipy.special import factorial
from scipy.optimize import curve_fit
from scipy.stats import chisquare
def reader(path):
    """
    This function gets a txt file, reads it, saves the first column "time tags",
     and returns them as an array.
    :param path: .txt file path
    :return: array of time tags
    """
    #read it
    df = pd.read_csv(path, comment='#', sep=';', header=None, skipinitialspace=True)

    # put the result in an array
    time_tags = df[0].to_numpy()
    return time_tags


def converter(time_tags):
    """
    this function converts time tags in a real second
    :param time_tags: array of time tags
    :return: array of time tags in seconds
    """

    time_tags_seconds = np.asarray(time_tags, dtype=float) * 80.955e-12
    return time_tags_seconds



def count_interval(time_tags_second,delta_interval = 1e-5):
    """"
    this function counts the numbers of clicks in different intervals of 10microseconds
    :param time_tags_second: array of time tags
    :return: number of clicks
    """

    time_tags_second = np.asarray(time_tags_second)

    if time_tags_second.size == 0:
        return np.array([], dtype=int)

    relative_times = time_tags_second - time_tags_second[0]
    interval_indices = np.floor(relative_times / delta_interval).astype(np.int64)

    counts = np.bincount(interval_indices)

    return counts


def number_of_clicks(count):

    """
    this function calculates the p(n) distribtions of clicks
    if i have for example 1000 invervals and in 100 of them are 2 i found 2 clicks the result is 10%
    formual is p(n) = (intervalls whit n clicks)/number of total intervals
    """

    count_intervals = np.asanyarray(count)
    if count_intervals.size == 0:
        return np.array([], dtype=float)
    number_of_intervals = np.bincount(count_intervals)

    p_n = number_of_intervals / count_intervals.size

    return p_n


def poisson_distribution(counts):
    counts = np.asarray(counts, dtype=int)
    if counts.size == 0:
        return np.array([], dtype=float)

    mean = np.mean(counts)
    n = np.arange(np.max(counts) + 1)

    p_poisson = np.exp(-mean) * mean ** n / factorial(n)

    return p_poisson


def bose_einstain_distribution(counts):

    counts = np.asarray(counts, dtype=int)

    if counts.size == 0:
        return np.array([], dtype=float)

    mean = np.mean(counts)
    n = np.arange(np.max(counts) + 1)

    p_be = (1 / (1 + mean)) * (mean / (1 + mean)) ** n

    return p_be


def avg_and_central_moments(counts):
    n = np.asarray(counts, dtype=float)

    if n.size == 0:
        raise ValueError("Array is empty")

    avg_clicks = np.mean(n)
    deviations = n - avg_clicks

    second_central_moment = np.mean(deviations ** 2)
    third_central_moment = np.mean(deviations ** 3)
    fourth_central_moment = np.mean(deviations ** 4)

    return (
        avg_clicks,
        second_central_moment,
        third_central_moment,
        fourth_central_moment
    )



def background_files(background_time_tags, duration):
    if duration <= 0:
        raise ValueError("the duration is not positive.")

    N = len(background_time_tags)
    R = N / duration

    return R


def background_mean(rate, delta_interval=1e-5):
    return rate * delta_interval


def fano_factor(counts):

    counts = np.asarray(counts, dtype=float)

    if counts.size == 0:
        raise ValueError("L'array dei conteggi è vuoto.")

    mean = np.mean(counts)

    if mean == 0:
        return np.nan

    variance = np.var(counts)

    return variance / mean




def fano_vs_bin_duration(time_tags_seconds, bin_durations, duration=None):
    durations = np.asarray(bin_durations, dtype=float)
    fano_values = []

    for dt in durations:
        counts = count_interval(time_tags_seconds, dt)
        fano_values.append(fano_factor(counts))

    return durations, np.asarray(fano_values)



def interarrival_times(time_tags_seconds):

    times = np.sort(np.asarray(time_tags_seconds, dtype=float))

    return np.diff(times)


def remove_afterpulses(time_tags_seconds, threshold=200e-9):
    """
    Rimuove i click separati da meno di threshold
    dall'ultimo click mantenuto.
    """
    times = np.sort(np.asarray(time_tags_seconds, dtype=float))

    if times.size == 0:
        return times

    filtered = [times[0]]

    for t in times[1:]:
        if t - filtered[-1] >= threshold:
            filtered.append(t)

    return np.asarray(filtered)



def poisson_moments(mu):
    mu2 = mu
    mu3 = mu
    mu4 = mu + 3 * mu**2

    return mu2, mu3, mu4


def bose_einstein_moments(mu):
    mu2 = mu + mu**2
    mu3 = mu + 3 * mu**2 + 2 * mu**3
    mu4 = mu + 10 * mu**2 + 18 * mu**3 + 9 * mu**4

    return mu2, mu3, mu4


def skewness_and_kurtosis(counts):

    counts = np.asarray(counts, dtype=float)

    mean = np.mean(counts)

    dev = counts - mean

    mu2 = np.mean(dev**2)
    mu3 = np.mean(dev**3)
    mu4 = np.mean(dev**4)

    if mu2 == 0:
        return np.nan, np.nan

    skewness = mu3 / mu2 ** 1.5
    kurtosis = mu4 / mu2 ** 2
    return skewness, kurtosis

def exponential(x, A, tau):
    return A * np.exp(-x / tau)

def fit_interarrival_distribution(time_tags_seconds):

    intervals = interarrival_times(
        time_tags_seconds
    )

    intervals = intervals[intervals > 200e-9]

    if len(intervals) < 10:
        raise ValueError(
            "Not enough intervals for exponential fit."
        )

    hist, edges = np.histogram(
        intervals,
        bins=100
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    popt, _ = curve_fit(
        exponential,
        centers,
        hist,
        p0=[
            np.max(hist),
            np.mean(intervals)
        ]
    )

    return popt

def generate_random_bits(time_tags_seconds,
                         threshold=200e-9):

    intervals = interarrival_times(
        time_tags_seconds
    )

    intervals = intervals[
        intervals >= threshold
    ]

    bits = []

    i = 0

    while i < len(intervals) - 1:

        delta1 = intervals[i]
        delta2 = intervals[i + 1]

        if delta1 > delta2:
            bits.append(1)
        else:
            bits.append(0)

        i += 2

    return np.asarray(bits, dtype=np.uint8)


def bits_to_bytes(bits):

    bits = np.asarray(
        bits,
        dtype=np.uint8
    )

    n_bytes = len(bits) // 8

    bytes_array = []

    for i in range(n_bytes):

        byte_bits = bits[
            i * 8:(i + 1) * 8
        ]

        value = 0

        for bit in byte_bits:
            value = (
                value << 1
            ) | bit

        bytes_array.append(value)

    return np.asarray(
        bytes_array,
        dtype=np.uint8
    )
def chi_square_test(byte_stream):

    hist, _ = np.histogram(
        byte_stream,
        bins=256,
        range=(0, 256)
    )

    expected = np.full(
        256,
        np.mean(hist)
    )

    chi2, p_value = chisquare(
        hist,
        expected
    )

    return chi2, p_value


def byte_mean(byte_stream):

    return np.mean(byte_stream)


def serial_correlation(byte_stream):

    x = byte_stream[:-1]
    y = byte_stream[1:]

    return np.corrcoef(
        x,
        y
    )[0, 1]


def bit_balance(bits):

    bits = np.asarray(bits)

    zeros = np.sum(bits == 0)
    ones = np.sum(bits == 1)

    return zeros, ones



def monte_carlo_pi(byte_stream):

    byte_stream = np.asarray(
        byte_stream,
        dtype=np.uint8
    )

    n_points = len(byte_stream) // 6

    if n_points == 0:
        return np.nan

    hits = 0

    for i in range(n_points):

        block = byte_stream[
            6 * i:6 * (i + 1)
        ].astype(np.int64)

        x = (
            block[0]
            + block[1]*256
            + block[2]*256**2
        )

        y = (
            block[3]
            + block[4]*256
            + block[5]*256**2
        )

        max_coord = 256**3 - 1

        x /= max_coord
        y /= max_coord

        if x*x + y*y <= 1:
            hits += 1

    return 4 * hits / n_points

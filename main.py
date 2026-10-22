import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # supaya bisa jalan tanpa GUI / simpan ke file
import matplotlib.pyplot as plt

TICKERS = {"BBRI": "BBRI.JK", "BMRI": "BMRI.JK"}
START = "2020-01-01"
TRADING_DAYS = 252             # jumlah hari bursa dalam setahun
RISK_FREE_ANNUAL = 0.06        # asumsi risk-free rate Indonesia ~6% per tahun


# =====================================================================
# LANGKAH 1: Download data
# =====================================================================
def download_data(ticker, start=START):
    """Ambil data harian dari Yahoo Finance, kembalikan DataFrame kolom Close."""
    import yfinance as yf
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(f"Data kosong untuk {ticker}")

    # yfinance versi baru mengembalikan MultiIndex kolom -> ratakan
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Close"]].dropna()


def generate_dummy(ticker, start=START, seed=0):
    """
    Fallback: buat data simulasi kalau internet tidak tersedia.
    Bukan data asli — hanya agar program tetap bisa dijalankan & diuji.
    """
    rng = np.random.default_rng(seed)
    tanggal = pd.bdate_range(start=start, end="2024-12-31")
    harga_awal = 4000 if "BBRI" in ticker else 6500
    # random walk dengan drift kecil
    ret = rng.normal(0.0004, 0.018, len(tanggal))
    harga = harga_awal * np.exp(np.cumsum(ret))
    return pd.DataFrame({"Close": harga}, index=tanggal)


def ambil_data(nama, ticker, seed):
    try:
        df = download_data(ticker)
        print(f"[OK]   {nama}: data asli dari yfinance ({len(df)} baris)")
        return df, True
    except Exception as e:
        print(f"[WARN] {nama}: gagal download ({type(e).__name__}). Pakai data simulasi.")
        return generate_dummy(ticker, seed=seed), False


# =====================================================================
# LANGKAH 2: Hitung return harian
# =====================================================================
def hitung_return(df):
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["cum_return"] = (1 + df["return"].fillna(0)).cumprod() - 1
    return df


# =====================================================================
# LANGKAH 3: Statistik
# =====================================================================
def hitung_statistik(df, nama, risk_free=RISK_FREE_ANNUAL):
    r = df["return"].dropna()

    mean_harian = r.mean()
    std_harian = r.std()

    # Annualisasi
    mean_tahunan = mean_harian * TRADING_DAYS
    std_tahunan = std_harian * np.sqrt(TRADING_DAYS)

    # Sharpe ratio sederhana = (return tahunan - risk free) / volatilitas tahunan
    sharpe = (mean_tahunan - risk_free) / std_tahunan

    total_return = df["cum_return"].iloc[-1]

    return {
        "saham": nama,
        "mean_return_harian": mean_harian,
        "std_return_harian": std_harian,
        "mean_return_tahunan": mean_tahunan,
        "volatilitas_tahunan": std_tahunan,
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "jumlah_hari": len(r),
    }


def cetak_statistik(stats):
    print("\n" + "=" * 60)
    print("STATISTIK RETURN")
    print("=" * 60)
    for s in stats:
        print(f"\n{s['saham']}  ({s['jumlah_hari']} hari bursa)")
        print(f"  Mean return harian    : {s['mean_return_harian']*100:.4f}%")
        print(f"  Std dev harian        : {s['std_return_harian']*100:.4f}%")
        print(f"  Mean return tahunan   : {s['mean_return_tahunan']*100:.2f}%")
        print(f"  Volatilitas tahunan   : {s['volatilitas_tahunan']*100:.2f}%")
        print(f"  Sharpe ratio          : {s['sharpe_ratio']:.3f}")
        print(f"  Total return periode  : {s['total_return']*100:.2f}%")

    terbaik_sharpe = max(stats, key=lambda s: s["sharpe_ratio"])
    terbaik_return = max(stats, key=lambda s: s["total_return"])

    print("\n" + "-" * 60)
    print(f"Sharpe ratio tertinggi (risk-adjusted): {terbaik_sharpe['saham']} "
          f"({terbaik_sharpe['sharpe_ratio']:.3f})")
    print(f"Total return tertinggi (absolut)      : {terbaik_return['saham']} "
          f"({terbaik_return['total_return']*100:.2f}%)")
    print("-" * 60)


# =====================================================================
# LANGKAH 4: Plot
# =====================================================================
def buat_plot(data, outfile="output/analisis_bbri_bmri.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    # -- Grafik 1: Perbandingan harga --
    for nama, df in data.items():
        ax1.plot(df.index, df["Close"], label=nama, linewidth=1.3)
    ax1.set_title("Perbandingan Harga Penutupan: BBRI vs BMRI")
    ax1.set_ylabel("Harga (Rp)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # -- Grafik 2: Cumulative return --
    for nama, df in data.items():
        ax2.plot(df.index, df["cum_return"] * 100, label=nama, linewidth=1.3)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Cumulative Return: BBRI vs BMRI")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_xlabel("Tanggal")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    print(f"\n[OK] Grafik disimpan: {outfile}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    data = {}
    for i, (nama, ticker) in enumerate(TICKERS.items()):
        df, _ = ambil_data(nama, ticker, seed=i)
        data[nama] = hitung_return(df)

    stats = [hitung_statistik(df, nama) for nama, df in data.items()]
    cetak_statistik(stats)

    buat_plot(data)

    # Simpan ringkasan ke CSV (opsional, berguna untuk README di GitHub)
    pd.DataFrame(stats).to_csv("output/ringkasan_statistik.csv", index=False)
    print("[OK] Ringkasan disimpan: ringkasan_statistik.csv")
    print(data['BMRI'])

if __name__ == "__main__":
    main()

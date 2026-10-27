import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Communication Systems FA2 Lab", page_icon="📡", layout="wide")

st.title("Communication Systems FA2 Simulation Lab")
st.caption("Activity 1: 8-QAM BER analysis  |  Activity 2: Multi-user OFDMA resource allocation")
st.info("Educational software simulation only. The OFDMA activity models a wireless scheduler; it does not connect to a real 5G network.")

# ---------- Shared helpers ----------
def gray2_to_index(bits2):
    """Gray labels 00, 01, 11, 10 map to amplitude indices 0, 1, 2, 3."""
    labels = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    return np.array([labels[(int(a), int(b))] for a, b in bits2], dtype=int)

def index_to_gray2(indices):
    labels = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=int)
    return labels[indices]

def qam8_mod(bits):
    """Rectangular 8-QAM: 4 x-levels and 2 y-levels; 3 bits/symbol."""
    bits = np.asarray(bits, dtype=int).reshape(-1, 3)
    xi = gray2_to_index(bits[:, :2])
    yi = bits[:, 2]
    x_levels = np.array([-3, -1, 1, 3], dtype=float)
    y_levels = np.array([-1, 1], dtype=float)
    # Average unnormalized symbol energy is 6; normalize Es to 1.
    return (x_levels[xi] + 1j * y_levels[yi]) / np.sqrt(6.0)

def qam8_demod(symbols):
    x_levels = np.array([-3, -1, 1, 3], dtype=float) / np.sqrt(6.0)
    y_levels = np.array([-1, 1], dtype=float) / np.sqrt(6.0)
    xi = np.argmin(np.abs(symbols.real[:, None] - x_levels[None, :]), axis=1)
    yi = np.argmin(np.abs(symbols.imag[:, None] - y_levels[None, :]), axis=1)
    xb = index_to_gray2(xi)
    return np.column_stack([xb, yi]).astype(int).reshape(-1)

def simulate_ber_8qam(ebn0_db, n_symbols, rng):
    bits = rng.integers(0, 2, size=n_symbols * 3)
    tx = qam8_mod(bits)
    ebn0 = 10 ** (ebn0_db / 10.0)
    # Es = 1 and k=3 bits/symbol => N0 = Es/(k*Eb/N0).
    n0 = 1.0 / (3.0 * ebn0)
    noise = np.sqrt(n0 / 2.0) * (
        rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
    )
    rx = tx + noise
    detected = qam8_demod(rx)
    errors = int(np.count_nonzero(bits != detected))
    return errors / bits.size, errors, bits.size, tx, rx

def qpsk_mod(bits):
    pairs = np.asarray(bits, dtype=int).reshape(-1, 2)
    return ((1 - 2 * pairs[:, 0]) + 1j * (1 - 2 * pairs[:, 1])) / np.sqrt(2.0)

def qpsk_demod(symbols):
    return np.column_stack([(symbols.real < 0).astype(int),
                            (symbols.imag < 0).astype(int)]).reshape(-1)

def simulate_ofdma(n_users, n_subcarriers, cp_len, snr_db, frames, allocation_mode, seed):
    rng = np.random.default_rng(seed)
    # Use only the first n_subcarriers FFT bins, split among users.
    bins = np.arange(n_subcarriers)
    if allocation_mode == "Contiguous groups":
        groups = np.array_split(bins, n_users)
    else:
        groups = [bins[u::n_users] for u in range(n_users)]

    user_bits = [0] * n_users
    user_errors = [0] * n_users
    user_symbols = [0] * n_users
    total_time_samples = []
    allocation = np.zeros((n_users, n_subcarriers), dtype=int)

    for u, g in enumerate(groups):
        allocation[u, g] = 1

    n0 = 10 ** (-snr_db / 10.0)  # QPSK Es=1 and Eb=Es/2, so this is a simplified SNR-based model.

    for _ in range(frames):
        freq = np.zeros(n_subcarriers, dtype=complex)
        tx_bits_by_user = []
        for u, g in enumerate(groups):
            b = rng.integers(0, 2, size=2 * len(g))
            s = qpsk_mod(b)
            freq[g] = s
            tx_bits_by_user.append(b)
            user_bits[u] += b.size
            user_symbols[u] += len(g)

        time_signal = np.fft.ifft(freq) * np.sqrt(n_subcarriers)
        cp = time_signal[-cp_len:] if cp_len > 0 else np.array([], dtype=complex)
        tx_with_cp = np.concatenate([cp, time_signal])
        noise = np.sqrt(n0 / 2.0) * (
            rng.standard_normal(tx_with_cp.size) + 1j * rng.standard_normal(tx_with_cp.size)
        )
        rx_with_cp = tx_with_cp + noise
        rx_time = rx_with_cp[cp_len:] if cp_len > 0 else rx_with_cp
        rx_freq = np.fft.fft(rx_time) / np.sqrt(n_subcarriers)

        if len(total_time_samples) < 1:
            total_time_samples = tx_with_cp[:min(120, tx_with_cp.size)]

        for u, g in enumerate(groups):
            detected = qpsk_demod(rx_freq[g])
            user_errors[u] += int(np.count_nonzero(tx_bits_by_user[u] != detected))

    ber = [user_errors[u] / max(user_bits[u], 1) for u in range(n_users)]
    # Relative throughput proxy: correctly received bits per simulated frame.
    throughput = [(user_bits[u] - user_errors[u]) / max(frames, 1) for u in range(n_users)]
    return allocation, np.array(total_time_samples), np.array(ber), np.array(throughput), groups

tab1, tab2 = st.tabs(["Activity 1 — 8-QAM BER", "Activity 2 — OFDMA application"])

# ---------- Activity 1 ----------
with tab1:
    st.header("Bit Error Rate (BER) analysis for 8-QAM")
    st.write("This activity sends random bits through an Additive White Gaussian Noise (AWGN) channel, detects the received symbols, and counts bit errors.")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_db = st.slider("Minimum Eb/N₀ (dB)", 0, 12, 0)
    with c2:
        max_db = st.slider("Maximum Eb/N₀ (dB)", 6, 24, 18)
    with c3:
        n_symbols = st.select_slider("Symbols per SNR point", options=[1000, 2000, 5000, 10000, 20000, 50000], value=10000)
    seed1 = st.number_input("Random seed (repeatable results)", min_value=0, max_value=999999, value=42, step=1, key="seed_ber")
    if max_db <= min_db:
        st.warning("Maximum Eb/N₀ must be greater than the minimum.")
    else:
        if st.button("Run 8-QAM BER simulation", type="primary", key="run_ber"):
            rng = np.random.default_rng(int(seed1))
            snr_values = np.arange(min_db, max_db + 1, 2)
            bers, errors_list, bits_list = [], [], []
            last_tx = last_rx = None
            progress = st.progress(0)
            for i, db in enumerate(snr_values):
                ber, errors, nbits, tx, rx = simulate_ber_8qam(float(db), int(n_symbols), rng)
                bers.append(ber)
                errors_list.append(errors)
                bits_list.append(nbits)
                if i == len(snr_values) - 1:
                    last_tx, last_rx = tx, rx
                progress.progress((i + 1) / len(snr_values))
            st.session_state["ber_results"] = {
                "snr": snr_values, "ber": np.array(bers), "errors": errors_list,
                "nbits": bits_list, "tx": last_tx, "rx": last_rx
            }
        if "ber_results" in st.session_state:
            r = st.session_state["ber_results"]
            left, right = st.columns(2)
            with left:
                fig, ax = plt.subplots()
                positive = np.maximum(r["ber"], 0.5 / np.array(r["nbits"]))
                ax.semilogy(r["snr"], positive, "o-", label="Simulated BER")
                ax.set_xlabel("Eb/N₀ (dB)")
                ax.set_ylabel("Bit Error Rate (BER)")
                ax.set_title("8-QAM BER over AWGN")
                ax.grid(True, which="both")
                ax.legend()
                st.pyplot(fig)
                st.caption("If zero errors occur, the graph displays a small plotting floor; the measured BER remains zero.")
            with right:
                fig2, ax2 = plt.subplots()
                ax2.scatter(r["tx"][:1500].real, r["tx"][:1500].imag, s=18, label="Transmitted", marker="o")
                ax2.scatter(r["rx"][:1500].real, r["rx"][:1500].imag, s=12, alpha=0.45, label="Received")
                ax2.set_xlabel("In-phase (I)")
                ax2.set_ylabel("Quadrature (Q)")
                ax2.set_title("8-QAM constellation (last SNR point)")
                ax2.grid(True)
                ax2.axis("equal")
                ax2.legend()
                st.pyplot(fig2)
            table_rows = []
            for db, ber, err, nb in zip(r["snr"], r["ber"], r["errors"], r["nbits"]):
                table_rows.append({"Eb/N₀ (dB)": int(db), "Bit errors": int(err), "Total bits": int(nb), "Measured BER": f"{ber:.6g}"})
            st.subheader("Measured results")
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
            st.success("Expected trend: BER generally decreases as Eb/N₀ increases. Because this is a random finite-length simulation, points may not decrease perfectly at every step.")
    with st.expander("What the blocks mean"):
        st.markdown("""
        **Random bits → 8-QAM modulator → AWGN channel → 8-QAM detector → error counter → BER graph**

        - **8-QAM:** eight constellation points; each symbol represents 3 bits.
        - **AWGN:** a simplified model of random channel noise.
        - **Eb/N₀:** bit energy relative to noise spectral density, shown in dB.
        - **BER:** number of incorrect detected bits divided by the number of transmitted bits.
        """)

# ---------- Activity 2 ----------
with tab2:
    st.header("Real-time-style OFDMA: multi-user wireless resource allocation")
    st.write("Scenario: a base station serves several users (video, video call, download, browsing) by assigning each user a group of orthogonal subcarriers.")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_users = st.selectbox("Number of users", [2, 3, 4, 5, 8], index=2)
    with c2:
        n_subcarriers = st.selectbox("Total subcarriers (FFT size)", [32, 64, 128, 256], index=1)
    with c3:
        allocation_mode = st.selectbox("Allocation strategy", ["Contiguous groups", "Round-robin groups"])
    c4, c5, c6 = st.columns(3)
    with c4:
        snr_db = st.slider("Channel SNR (dB)", 0, 30, 18, key="ofdma_snr")
    with c5:
        cp_len = st.select_slider("Cyclic prefix length (samples)", options=[0, 4, 8, 16], value=8)
    with c6:
        frames = st.select_slider("Frames to simulate", options=[5, 10, 25, 50, 100], value=25)
    seed2 = st.number_input("Random seed (repeatable results)", min_value=0, max_value=999999, value=7, step=1, key="seed_ofdma")
    if st.button("Run OFDMA simulation", type="primary", key="run_ofdma"):
        alloc, wave, ber, thr, groups = simulate_ofdma(
            int(n_users), int(n_subcarriers), int(cp_len), float(snr_db),
            int(frames), allocation_mode, int(seed2)
        )
        st.session_state["ofdma_results"] = {
            "alloc": alloc, "wave": wave, "ber": ber, "thr": thr,
            "groups": [list(map(int, g)) for g in groups],
            "n_users": int(n_users), "n_subcarriers": int(n_subcarriers),
            "mode": allocation_mode, "snr": int(snr_db), "frames": int(frames)
        }
    if "ofdma_results" in st.session_state:
        r2 = st.session_state["ofdma_results"]
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.imshow(r2["alloc"], aspect="auto", interpolation="nearest", origin="lower")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("User")
            ax.set_yticks(range(r2["n_users"]))
            ax.set_yticklabels([f"User {i+1}" for i in range(r2["n_users"])])
            ax.set_title("OFDMA resource allocation map (1 = assigned)")
            ax.set_xticks(np.arange(0, r2["n_subcarriers"], max(1, r2["n_subcarriers"] // 8)))
            ax.grid(False)
            st.pyplot(fig)
        with right:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(r2["wave"])), np.real(r2["wave"]), label="In-phase")
            ax.plot(np.arange(len(r2["wave"])), np.imag(r2["wave"]), label="Quadrature", alpha=0.8)
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Amplitude")
            ax.set_title("Example OFDM time-domain waveform")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        left2, right2 = st.columns(2)
        with left2:
            fig, ax = plt.subplots()
            ax.bar([f"User {i+1}" for i in range(r2["n_users"])], r2["ber"])
            ax.set_ylabel("BER")
            ax.set_title("BER by user")
            ax.grid(True, axis="y")
            st.pyplot(fig)
        with right2:
            fig, ax = plt.subplots()
            ax.bar([f"User {i+1}" for i in range(r2["n_users"])], r2["thr"])
            ax.set_ylabel("Correct bits per simulated frame")
            ax.set_title("Successful data delivery (throughput proxy)")
            ax.grid(True, axis="y")
            st.pyplot(fig)
        st.subheader("Resource assignment")
        assignment_rows = []
        for i, g in enumerate(r2["groups"]):
            assignment_rows.append({
                "User": f"User {i+1}",
                "Allocated subcarriers": len(g),
                "Subcarrier indices": ", ".join(map(str, g))
            })
        st.dataframe(assignment_rows, use_container_width=True, hide_index=True)
        st.caption(f"Configuration: {r2['n_users']} users, {r2['n_subcarriers']} subcarriers, {r2['mode']}, SNR={r2['snr']} dB, {r2['frames']} frames.")
        st.success("Expected trend: higher SNR generally reduces BER. Users receive distinct subcarrier groups in the same OFDM symbol, illustrating OFDMA. Throughput values are a simplified simulation metric, not measured real-world network throughput.")
    with st.expander("What the blocks mean"):
        st.markdown("""
        **User data → QPSK symbols → subcarrier scheduler → IFFT → cyclic prefix → noisy channel → remove prefix → FFT → user-wise detection**

        - **OFDMA scheduler:** assigns different subcarrier indices to different users.
        - **IFFT:** combines frequency-domain subcarriers into a time-domain OFDM waveform.
        - **Cyclic prefix (CP):** a copied end section added before transmission to help handle multipath delay; this demo uses an AWGN-only channel.
        - **FFT:** recovers the subcarrier values at the receiver.
        - **BER by user:** checks each user's recovered bits.
        """)

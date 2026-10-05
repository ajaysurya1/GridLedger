"""
GridLedger voltage cross-check module (Phase 3).

Cross-validates non-technical loss (NTL) anomalies with low-voltage (LV) network physics.
Compares digital-twin simulated voltage (derived from topology and smart-meter power)
against measured smart-meter voltage. The daily minimum voltage deviation is the most
informative indicator for persistent unmetered load.

Pure detection functions only — NO ground-truth oracle imports.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from gridledger.config import CFG, Config


def build_Z(parent: np.ndarray, r_seg: np.ndarray) -> np.ndarray:
    """
    Build the NxN bus impedance (resistance) matrix for a radial LV tree.
    Z[i, j] is the total resistance of the path shared by nodes i and j from root (node 0).
    parent[k] < k is assumed for all k >= 1; parent[0] = 0.
    """
    N = len(parent)
    Z = np.zeros((N, N), dtype=np.float64)

    # Find ancestor path (sequence of nodes from root to k) for each node
    anc: List[List[int]] = [[] for _ in range(N)]
    for k in range(1, N):
        curr = k
        path = []
        while curr > 0:
            path.append(curr)
            curr = int(parent[curr])
        anc[k] = path

    for i in range(N):
        anc_i_set = set(anc[i])
        for j in range(i, N):
            if i == 0 or j == 0:
                Z[i, j] = Z[j, i] = 0.0
            else:
                # Common path segments
                common = anc_i_set.intersection(anc[j])
                val = float(np.sum(r_seg[list(common)]))
                Z[i, j] = Z[j, i] = val

    return Z


def node_voltage(
    Z: np.ndarray,
    P_w: np.ndarray,
    v0: np.ndarray | float,
) -> np.ndarray:
    """
    First-order linearized voltage drop at all LV nodes.
    V = v0 - (Z @ P_w) / v0
    P_w: (N, T) power in Watts at each node (or 1D array (N,))
    v0: (T,) or scalar secondary voltage in Volts
    """
    if P_w.ndim == 1:
        drop = (Z @ P_w) / float(v0)
        return float(v0) - drop
    
    if isinstance(v0, (int, float)):
        v0_arr = np.full(P_w.shape[1], float(v0), dtype=np.float64)
    else:
        v0_arr = np.asarray(v0, dtype=np.float64)

    drop = (Z @ P_w) / v0_arr[None, :]
    return v0_arr[None, :] - drop


def daily_min_voltage(V: np.ndarray, n_days: int) -> np.ndarray:
    """
    Compute daily minimum voltage for each node/customer, ignoring NaNs.
    V: (n_nodes, T) where T = n_days * 24.
    Returns: (n_nodes, n_days) array.
    """
    n_nodes, T = V.shape
    assert T == n_days * 24, f"Mismatch: T={T} vs {n_days}*24"
    V_reshaped = V.reshape(n_nodes, n_days, 24)
    
    # Nanmin over hour axis (axis=2)
    # Suppress all-NaN slice warnings by checking nan count
    all_nan = np.all(np.isnan(V_reshaped), axis=2)
    min_v = np.nanmin(V_reshaped, axis=2)
    min_v[all_nan] = np.nan
    return min_v


def calibrate_dv(
    dv_meas_sim: np.ndarray,
    cal_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate baseline per-meter voltage bias and variance during burn-in period.
    dv_meas_sim = Measured_min - Simulated_min (shape: n_cust, n_days).
    Under unmetered theft, Measured < Simulated => dv_meas_sim is negative.
    
    Returns:
        bias: (n_cust,) median bias during cal_days
        sigma: (n_cust,) robust MAD std dev during cal_days (floored at 0.05 V)
        z_scores: (n_cust, n_days) standardised drop anomaly (positive = measured is abnormally low)
    """
    n_cust, n_days = dv_meas_sim.shape
    cal_window = dv_meas_sim[:, :cal_days]
    
    bias = np.nanmedian(cal_window, axis=1)
    bias = np.where(np.isnan(bias), 0.0, bias)

    residuals = cal_window - bias[:, None]
    mad = np.nanmedian(np.abs(residuals), axis=1)
    sigma = 1.4826 * mad
    sigma = np.where(np.isnan(sigma) | (sigma < 0.05), 0.10, sigma)

    # Positive z indicates Measured voltage is lower than expected
    diff = dv_meas_sim - bias[:, None]
    z_scores = -diff / sigma[:, None]
    
    return bias, sigma, z_scores


def localize_extra_load(
    Z_model: np.ndarray,
    meter_nodes: np.ndarray,
    cand_nodes: np.ndarray,
    dv_volts: np.ndarray,
    v0_nominal: float = 230.0,
    min_alpha_w: float = 300.0,
) -> Tuple[int, float, float]:
    """
    Localize unmetered load location among candidate nodes via least-squares matching
    against the LV impedance matrix.
    
    dv_volts: (n_meters,) observed voltage deficit (Simulated - Measured) at valid meters.
    cand_nodes: list of candidate node indices in the LV tree (poles + customer nodes).
    
    Returns:
        best_node: candidate node index with best match
        best_alpha_w: estimated stolen power in Watts
        best_r2: goodness of fit R^2 in [0, 1]
    """
    valid = ~np.isnan(dv_volts) & (dv_volts > 0.0)
    if np.sum(valid) < 3:
        # Not enough meters with positive drop to localize reliably
        return int(cand_nodes[0]), 0.0, 0.0

    dv_v = dv_volts[valid]
    valid_meters = meter_nodes[valid]
    
    ss_tot = float(np.sum((dv_v - np.mean(dv_v)) ** 2))
    if ss_tot < 1e-6:
        ss_tot = 1e-6

    best_node = int(cand_nodes[0])
    best_alpha = 0.0
    best_r2 = 0.0

    for node in cand_nodes:
        # Theoretical sensitivity h_i = Z_model[m_i, node] / v0
        h = Z_model[valid_meters, node] / v0_nominal
        h_norm = float(np.sum(h ** 2))
        if h_norm < 1e-9:
            continue

        # Least squares estimate of extra power alpha (in Watts)
        alpha = float(np.sum(h * dv_v) / h_norm)
        if alpha < 0.0:
            continue

        pred = h * alpha
        ss_res = float(np.sum((dv_v - pred) ** 2))
        r2 = 1.0 - (ss_res / ss_tot)
        r2 = max(0.0, min(1.0, r2))

        if r2 > best_r2:
            best_r2 = r2
            best_node = int(node)
            best_alpha = alpha

    return best_node, best_alpha, best_r2


def _rng_lv(seed: int, key: str) -> np.random.Generator:
    """Deterministic RNG derived from seed and string key."""
    h = hash(key) & 0xFFFFFFFF
    return np.random.default_rng(seed ^ h)


def build_lv_network(
    tx_idx: int,
    seed: int,
    customer_meter_indices: np.ndarray,
    cfg: Config = CFG,
) -> Dict[str, Any]:
    """
    Synthesize realistic single-phase-equivalent LV network topology for a transformer.
    - 2-3 laterals originating from node 0 (transformer secondary)
    - 4-6 poles per lateral spaced 30-50m apart
    - 1-3 customers tapped per pole via service drops (10-25m)
    """
    rng = _rng_lv(seed, f"tx_lv_{tx_idx}")
    n_cust = len(customer_meter_indices)

    # Topology dimensions
    n_laterals = rng.integers(2, 4)
    poles_per_lateral = rng.integers(4, 7, size=n_laterals)
    total_poles = int(np.sum(poles_per_lateral))

    # Node indices:
    # 0 = Transformer secondary
    # 1 .. total_poles = Pole nodes
    # total_poles + 1 .. total_poles + n_cust = Customer nodes
    N = 1 + total_poles + n_cust
    parent = np.zeros(N, dtype=np.int32)
    dist_m = np.zeros(N, dtype=np.float64)
    pole_nodes: List[int] = []

    current_pole_idx = 1
    for lat_i in range(n_laterals):
        n_p = poles_per_lateral[lat_i]
        prev_node = 0  # Starts from transformer
        for p in range(n_p):
            pole_id = current_pole_idx
            parent[pole_id] = prev_node
            # 30-50m pole-to-pole span
            span = rng.uniform(30.0, 50.0) if prev_node > 0 else rng.uniform(20.0, 40.0)
            dist_m[pole_id] = span
            pole_nodes.append(pole_id)
            prev_node = pole_id
            current_pole_idx += 1

    pole_nodes_arr = np.array(pole_nodes, dtype=np.int32)

    # Assign customers to poles uniformly
    cust_nodes = np.arange(1 + total_poles, N, dtype=np.int32)
    assigned_poles = rng.choice(pole_nodes_arr, size=n_cust, replace=True)
    
    for c_i, c_node in enumerate(cust_nodes):
        p_node = assigned_poles[c_i]
        parent[c_node] = p_node
        # Service drop length: 10-25m
        dist_m[c_node] = rng.uniform(10.0, 25.0)

    # Resistance per meter:
    # Main trunk / lateral: AAC/ACSR ~ 0.0004 - 0.0006 ohm/m
    # Service drop: Service cable ~ 0.0010 - 0.0018 ohm/m
    r_seg_true = np.zeros(N, dtype=np.float64)
    for k in range(1, N):
        if k <= total_poles:
            r_per_m = rng.uniform(0.00045, 0.00065)
        else:
            r_per_m = rng.uniform(0.00110, 0.00160)
        r_seg_true[k] = dist_m[k] * r_per_m

    # Digital twin utility model has lognormal parameter error (per segment)
    lognorm_err = rng.lognormal(0.0, cfg.lv_model_lognorm_sd, size=N)
    r_seg_model = r_seg_true * lognorm_err
    r_seg_model[0] = 0.0

    Z_true = build_Z(parent, r_seg_true)
    Z_model = build_Z(parent, r_seg_model)

    return {
        "N": N,
        "parent": parent,
        "dist_m": dist_m,
        "pole_nodes": pole_nodes_arr,
        "cust_nodes": cust_nodes,
        "cust_meter": np.array(customer_meter_indices, dtype=np.int32),
        "r_seg_true": r_seg_true,
        "r_seg_model": r_seg_model,
        "Z_true": Z_true,
        "Z_model": Z_model,
        "n_laterals": n_laterals,
    }


def generate_voltage_measurements(
    lv: Dict[str, Any],
    cust_true_kwh: np.ndarray,
    seed: int,
    tx_idx: int,
    T: int,
    cfg: Config = CFG,
    tap_pole_node: Optional[int] = None,
    tap_kw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic physical voltages at LV customer meters.
    V_true = v0_t - (Z_true @ P_inst)/v0_t + noise
    Returns:
        v_meas: (n_cust, T) array with NaN where meter was not sampled (sample prob p=0.55)
        v0_t: (T,) transformer secondary voltage with slight noise
    """
    rng = _rng_lv(seed, f"tx_volt_meas_{tx_idx}")
    N = lv["N"]
    cust_nodes = lv["cust_nodes"]
    Z_true = lv["Z_true"]

    # Nominal 230V with slight substation fluctuation: 228 - 236V diurnal swing
    hour_of_day = np.arange(T) % 24
    diurnal_v0 = 232.0 + 3.0 * np.sin((hour_of_day - 6) * 2 * np.pi / 24)
    v0_true = diurnal_v0 + rng.normal(0.0, 0.4, size=T)
    v0_meas = v0_true + rng.normal(0.0, cfg.lv_v0_noise_v, size=T)

    # Node power matrix in Watts: (N, T)
    P_node = np.zeros((N, T), dtype=np.float64)
    # Convert kWh/h to Watts (1 kWh in 1 hour = 1000 W average)
    P_node[cust_nodes, :] = cust_true_kwh * 1000.0

    # Inject illegal tap at pole if present
    if tap_pole_node is not None and tap_kw is not None:
        P_node[tap_pole_node, :] += tap_kw * 1000.0

    # Compute physical true voltages
    V_all_true = node_voltage(Z_true, P_node, v0_true)  # (N, T)

    # Customer meters: add meter noise
    V_cust_true = V_all_true[cust_nodes, :]  # (n_cust, T)
    meas_noise = rng.normal(0.0, cfg.lv_meas_noise_v, size=V_cust_true.shape)
    V_cust_meas = V_cust_true + meas_noise

    # PLC / RF sampling: each customer-hour has sample with p = lv_sample_prob
    sampled = rng.uniform(0.0, 1.0, size=V_cust_meas.shape) < cfg.lv_sample_prob
    V_cust_sampled = np.where(sampled, V_cust_meas, np.nan)

    return V_cust_sampled.astype(np.float32), v0_meas.astype(np.float32)


def analyse_tx_voltage(
    tx_id: str,
    lv: Dict[str, Any],
    v_meas: np.ndarray,
    v0_t: np.ndarray,
    metered_kwh: np.ndarray,
    today_day: int,
    n_days: int = 60,
    cal_days: int = 14,
    cfg: Config = CFG,
) -> Dict[str, Any]:
    """
    Run digital twin voltage simulation and compare with measured voltages.
    NO oracle peeking: uses only days < today_day and metered customer power.
    
    Returns structured results including alarm status, z-scores, localized candidate node.
    """
    N = lv["N"]
    cust_nodes = lv["cust_nodes"]
    Z_model = lv["Z_model"]
    pole_nodes = lv["pole_nodes"]
    n_cust = len(cust_nodes)
    T = n_days * 24

    # Digital twin simulation using METERED consumption (what the utility knows)
    P_model = np.zeros((N, T), dtype=np.float64)
    # Fill metered kWh (up to today_day*24; zero after)
    T_known = min(T, today_day * 24)
    P_model[cust_nodes, :T_known] = np.nan_to_num(metered_kwh[:, :T_known], nan=0.0) * 1000.0

    # Simulated expected voltages
    V_sim_all = node_voltage(Z_model, P_model, v0_t)
    V_sim_cust = V_sim_all[cust_nodes, :]

    # Daily minimums
    daily_meas_min = daily_min_voltage(v_meas, n_days)  # (n_cust, n_days)
    daily_sim_min = daily_min_voltage(V_sim_cust, n_days)  # (n_cust, n_days)

    dv_meas_sim = daily_meas_min - daily_sim_min
    bias, sigma, z_scores = calibrate_dv(dv_meas_sim, cal_days)

    # Check recent window: [today_day - window_days, today_day)
    w_start = max(cal_days, today_day - cfg.volt_alarm_window_days)
    w_end = today_day

    alarm = False
    alarm_days: List[int] = []
    cand_node: Optional[int] = None
    cand_pole: Optional[int] = None
    cand_meter: Optional[int] = None
    alpha_kw: float = 0.0
    r2_fit: float = 0.0
    e_volt: float = 0.0

    if w_end > w_start:
        window_z = z_scores[:, w_start:w_end]  # (n_cust, w_len)
        # Check if any meter consistently triggers z > volt_z_alarm
        # or aggregate transformer max z-score
        max_z_per_day = np.nanmax(window_z, axis=0)  # (w_len,)
        exceed_days = np.where(max_z_per_day >= cfg.volt_z_alarm)[0]
        frac_exceed = len(exceed_days) / max(1, (w_end - w_start))

        if frac_exceed >= cfg.volt_alarm_day_frac:
            alarm = True
            alarm_days = [int(w_start + d) for d in exceed_days]

            # Use mean drop deficit during alarm days to localize
            recent_sim_min = np.nanmean(daily_sim_min[:, alarm_days], axis=1)
            recent_meas_min = np.nanmean(daily_meas_min[:, alarm_days], axis=1)
            dv_recent = recent_sim_min - recent_meas_min  # Positive when measured is lower

            # Candidate nodes: all poles + all customer nodes
            cand_pool = np.concatenate([pole_nodes, cust_nodes])
            best_node, alpha_w, r2 = localize_extra_load(
                Z_model,
                cust_nodes,
                cand_pool,
                dv_recent,
                v0_nominal=cfg.v_nominal,
                min_alpha_w=cfg.volt_alpha_min_w,
            )

            cand_node = best_node
            alpha_kw = alpha_w / 1000.0
            r2_fit = r2

            # Determine whether candidate is a pole or customer node
            if best_node in pole_nodes:
                cand_pole = int(best_node)
            else:
                cust_match = np.where(cust_nodes == best_node)[0]
                if len(cust_match) > 0:
                    cand_meter = int(lv["cust_meter"][cust_match[0]])

            # Evidence score e_volt in [0, 1]
            # e_volt = r2_fit * 1[alpha >= min_alpha]
            if alpha_w >= cfg.volt_alpha_min_w:
                e_volt = float(np.clip(r2_fit, 0.0, 1.0))

    mean_z_window = float(np.nanmean(z_scores[:, w_start:w_end])) if w_end > w_start else 0.0
    max_z_window = float(np.nanmax(z_scores[:, w_start:w_end])) if w_end > w_start else 0.0

    return {
        "tx_id": tx_id,
        "alarm": alarm,
        "alarm_days": alarm_days,
        "max_z": max_z_window,
        "mean_z": mean_z_window,
        "cand_node": cand_node,
        "cand_pole": cand_pole,
        "cand_meter": cand_meter,
        "alpha_kw": alpha_kw,
        "r2": r2_fit,
        "e_volt": e_volt,
        "daily_meas_min": daily_meas_min,
        "daily_sim_min": daily_sim_min,
        "z_scores": z_scores,
        "cal_sigma": sigma,
        "cal_bias": bias,
    }

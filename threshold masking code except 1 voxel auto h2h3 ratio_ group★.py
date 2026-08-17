#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, simpledialog, messagebox
from skimage import io
from skimage.measure import label
from numpy.polynomial import polynomial as poly


# =========================
# ⭐ 폴더명에서 registered 번호 추출
#   규칙: "3D_x_fit_3★" -> x 추출 -> registered{x}
# =========================
def extract_registered_number_from_star_folder(name: str):
    if "★" not in name:
        return None
    m = re.search(r"3D_(\d+)", name)
    return int(m.group(1)) if m else None


# =========================================================
# shape normalize 유틸
# =========================================================
def normalize_mask_to_shape(mask_2d, target_shape):
    out = np.zeros(target_shape, dtype=bool)
    h = min(mask_2d.shape[0], target_shape[0])
    w = min(mask_2d.shape[1], target_shape[1])
    out[:h, :w] = mask_2d[:h, :w]
    return out


def normalize_array_to_shape(arr_2d, target_shape, fill_value=np.nan):
    out = np.full(target_shape, fill_value, dtype=arr_2d.dtype)
    h = min(arr_2d.shape[0], target_shape[0])
    w = min(arr_2d.shape[1], target_shape[1])
    out[:h, :w] = arr_2d[:h, :w]
    return out


# =========================================================
# 여러 상위 폴더 연속 선택
# =========================================================
def ask_parent_directories():
    parents = []
    while True:
        folder = filedialog.askdirectory(
            title="상위 폴더 선택 (취소 누르면 선택 종료)"
        )
        if not folder:
            break

        if folder not in parents:
            parents.append(folder)

        more = messagebox.askyesno(
            "폴더 추가 선택",
            "다른 상위 폴더를 추가로 선택하시겠습니까?"
        )
        if not more:
            break

    return parents


# =========================================================
# phase fraction table 관련
# =========================================================
def ask_phase_fraction_excel():
    path = filedialog.askopenfilename(
        title="SOC-H2-H3 비율 엑셀 선택",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    if not path:
        raise ValueError("❌ phase fraction 엑셀을 선택하지 않았습니다.")
    return path


def ask_phase_selection():
    yes_h3 = messagebox.askyesno(
        "마스크 상 선택",
        "H3 마스크를 생성할까요?\n\nYES = H3 마스크\nNO = H2 마스크"
    )
    return "H3" if yes_h3 else "H2"


def ask_soc_reference_energies():
    s = simpledialog.askstring(
        "SOC 기준 에너지 입력",
        "SOC 변환 기준 peak-site 에너지를 입력하세요.\n"
        "형식: soc0_energy,soc100_energy\n"
        "예: 8353.6,8356.2\n\n"
        "의미:\n"
        "8353.6 -> SOC 0\n"
        "8356.2 -> SOC 100",
        initialvalue="8353.6,8356.2"
    )
    if s is None or not s.strip():
        raise ValueError("❌ SOC 기준 에너지를 입력하지 않았습니다.")

    try:
        a, b = s.replace(" ", "").split(",")
        e0 = float(a)
        e100 = float(b)
    except Exception:
        raise ValueError("❌ 형식 오류: soc0_energy,soc100_energy  예: 8353.6,8356.2")

    if e0 == e100:
        raise ValueError("❌ soc0_energy와 soc100_energy는 같을 수 없습니다.")

    return e0, e100


def load_phase_fraction_table(excel_path):
    df = pd.read_excel(excel_path)

    if df.shape[1] < 3:
        raise ValueError("❌ phase fraction 엑셀은 최소 3개 열(SOC, H2, H3)이 필요합니다.")

    cols_lower = [str(c).strip().lower() for c in df.columns]

    soc_col = None
    h2_col = None
    h3_col = None

    for c, cl in zip(df.columns, cols_lower):
        if soc_col is None and "soc" in cl:
            soc_col = c
        if h2_col is None and ("h2" in cl or "phase2" in cl):
            h2_col = c
        if h3_col is None and ("h3" in cl or "phase3" in cl):
            h3_col = c

    if soc_col is None:
        soc_col = df.columns[0]
    if h2_col is None:
        h2_col = df.columns[1]
    if h3_col is None:
        h3_col = df.columns[2]

    out = pd.DataFrame({
        "SOC": pd.to_numeric(df[soc_col], errors="coerce"),
        "H2": pd.to_numeric(df[h2_col], errors="coerce"),
        "H3": pd.to_numeric(df[h3_col], errors="coerce"),
    }).dropna()

    if out.empty:
        raise ValueError("❌ phase fraction 엑셀에서 유효한 SOC/H2/H3 데이터를 읽지 못했습니다.")

    out = out.sort_values("SOC").reset_index(drop=True)
    return out


def interpolate_h2_h3_from_soc(soc_value, frac_df):
    soc_arr = frac_df["SOC"].to_numpy(dtype=float)
    h2_arr = frac_df["H2"].to_numpy(dtype=float)
    h3_arr = frac_df["H3"].to_numpy(dtype=float)

    soc_min = float(np.min(soc_arr))
    soc_max = float(np.max(soc_arr))
    soc_clamped = min(max(float(soc_value), soc_min), soc_max)

    h2 = float(np.interp(soc_clamped, soc_arr, h2_arr))
    h3 = float(np.interp(soc_clamped, soc_arr, h3_arr))

    return soc_clamped, h2, h3


def percentile_range_from_phase_fraction(phase_name, h2_frac, h3_frac):
    """
    H3 = 상위 h3_frac
      -> top [0, h3*100]
    H2 = 하위 h2_frac
      -> top [h3*100, 100]
    """
    h2_frac = float(h2_frac)
    h3_frac = float(h3_frac)

    if h2_frac > 1.0 or h3_frac > 1.0:
        h2_frac /= 100.0
        h3_frac /= 100.0

    h2_frac = max(0.0, min(1.0, h2_frac))
    h3_frac = max(0.0, min(1.0, h3_frac))

    if phase_name.upper() == "H3":
        top_p_low = 0.0
        top_p_high = 100.0 * h3_frac
    else:  # H2
        top_p_low = 100.0 * h3_frac
        top_p_high = 100.0

    top_p_low = max(0.0, min(100.0, top_p_low))
    top_p_high = max(0.0, min(100.0, top_p_high))

    if top_p_low > top_p_high:
        top_p_low, top_p_high = top_p_high, top_p_low

    return top_p_low, top_p_high


# =========================================================
# SOC 자동 계산: roi_peak_summary.csv의 입자 평균 peak-site 사용
# =========================================================
def read_particle_mean_peak_from_roi_summary(reg_dir):
    csv_path = os.path.join(reg_dir, "roi_peak_summary.csv")
    if not os.path.exists(csv_path):
        raise ValueError(f"[X] roi_peak_summary.csv 없음: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1순위: Slice == average 인 행의 Mean Peak (eV)
    if "Slice" in df.columns and "Mean Peak (eV)" in df.columns:
        avg_rows = df[df["Slice"].astype(str).str.lower() == "average"]
        if len(avg_rows) > 0:
            val = pd.to_numeric(avg_rows["Mean Peak (eV)"], errors="coerce").dropna()
            if len(val) > 0:
                return float(val.iloc[0]), "roi_peak_summary:average_mean_peak"

    # 2순위: Particle Mean Peak (eV) 컬럼의 마지막 유효값
    if "Particle Mean Peak (eV)" in df.columns:
        val = pd.to_numeric(df["Particle Mean Peak (eV)"], errors="coerce").dropna()
        if len(val) > 0:
            return float(val.iloc[-1]), "roi_peak_summary:particle_mean_peak"

    raise ValueError(f"[X] roi_peak_summary.csv에서 particle mean peak를 찾지 못했습니다: {csv_path}")


def convert_peak_to_soc(mean_peak_eV, soc0_energy, soc100_energy):
    soc = 100.0 * (float(mean_peak_eV) - float(soc0_energy)) / (float(soc100_energy) - float(soc0_energy))
    return soc


def get_soc_for_registered(reg_dir, soc0_energy, soc100_energy):
    mean_peak_eV, source = read_particle_mean_peak_from_roi_summary(reg_dir)
    soc_value = convert_peak_to_soc(mean_peak_eV, soc0_energy, soc100_energy)
    return float(soc_value), source, float(mean_peak_eV)


# =========================================================
# (1) legacy-fit energy 축 정합
# =========================================================
def align_energy_axis_for_legacy_fit(img_stack, eng, energy_axis_reverse=True, verbose=False):
    eng = np.array(eng, dtype=float).ravel()

    if img_stack.ndim != 3:
        raise ValueError(f"[X] img_stack must be 3D (E,H,W). Got shape={img_stack.shape}")
    if img_stack.shape[0] != len(eng):
        raise ValueError(f"[X] energy length mismatch: img_stack E={img_stack.shape[0]} vs len(eng)={len(eng)}")

    if energy_axis_reverse:
        img_stack = img_stack[::-1, :, :]
        if verbose:
            print("[i] img_stack energy axis reversed: img_stack[::-1,:,:]")

    if eng[0] > eng[-1]:
        eng = eng[::-1]
        if verbose:
            print("[i] eng reversed to increasing order")

    diffs = np.diff(eng)
    if np.any(diffs <= 0):
        raise ValueError("[X] eng가 단조 증가가 아닙니다. energy.txt 순서/매칭 확인 필요.")
    return img_stack, eng


# =========================================================
# (2) 피크맵: legacy 피팅 유지
# =========================================================
def polynomial_second_fit_separate_raw(img_stack, eng, fit_num, ev_step):
    y = img_stack.reshape(len(img_stack), -1)
    y_new = np.zeros((fit_num * 2 + 1, y.shape[1]), dtype=np.float32)
    mp_list = []

    for i in range(y.shape[1]):
        a = int(np.argmax(y[:, i]))
        if a + fit_num + 1 > len(y):
            y_new[:, i] = y[len(y) - 2 * fit_num - 1:len(y) + 1, i]
        elif a - fit_num < 0:
            y_new[:, i] = y[0: 2 * fit_num + 1, i]
        else:
            y_new[:, i] = y[a - fit_num:a + fit_num + 1, i]
        mp_list.append(a)

    x = np.linspace(0, ev_step * (len(y_new) - 1), len(y_new))
    coefs, _ = poly.polyfit(x, y_new, 2, full=True)

    ind = [a - fit_num for a in mp_list]
    denom = 2.0 * coefs[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x0 = -(coefs[1] / denom) + eng[ind]
    x0[~np.isfinite(x0)] = np.nan

    peak_map = x0.reshape(img_stack.shape[1], img_stack.shape[2])
    return peak_map


def parse_slice_number_from_mask_filename(fn):
    m = re.search(r"roi_mask_slice_(\d+)\.npy$", fn)
    return int(m.group(1)) if m else None


# =========================================================
# 유효복셀 판정 + clip
# =========================================================
def filter_and_clip_valid_peak_values(values_1d, low, high, low_margin, high_margin):
    v = np.asarray(values_1d, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([], dtype=np.float32)

    eff_low = float(low) - float(low_margin)
    eff_high = float(high) + float(high_margin)

    valid = v[(v >= eff_low) & (v <= eff_high)]
    if valid.size == 0:
        return np.array([], dtype=np.float32)

    valid = valid.copy()
    valid[valid < low] = float(low)
    valid[valid > high] = float(high)
    return valid.astype(np.float32)


def top_based_thresholds(values_1d, top_p_low, top_p_high):
    v = np.asarray(values_1d, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan

    bottom_p_low = 100.0 - float(top_p_low)
    bottom_p_high = 100.0 - float(top_p_high)

    p_lo, p_hi = sorted([bottom_p_high, bottom_p_low])
    thr_lo, thr_hi = np.percentile(v, [p_lo, p_hi])
    return float(thr_lo), float(thr_hi)


# =========================================================
# 3D singleton voxel 제거
#   connectivity=1 -> 6-neighbour(face-connected)
# =========================================================
def remove_singleton_voxels_3d(mask_3d, connectivity=1):
    mask_3d = np.asarray(mask_3d, dtype=bool)
    if mask_3d.ndim != 3 or not np.any(mask_3d):
        return mask_3d.copy()

    lbl = label(mask_3d.astype(np.uint8), connectivity=connectivity)
    if lbl.max() == 0:
        return mask_3d.copy()

    counts = np.bincount(lbl.ravel())
    keep_labels = np.where(counts > 1)[0]
    keep_labels = keep_labels[keep_labels != 0]

    if keep_labels.size == 0:
        return np.zeros_like(mask_3d, dtype=bool)

    cleaned = np.isin(lbl, keep_labels)
    return cleaned


# =========================================================
# registered 1개 처리
# =========================================================
def process_registered_make_filtered_masks(
    reg_dir,
    mask_dir,
    eng,
    ev_step,
    energy_axis_reverse,
    low,
    high,
    low_margin,
    high_margin,
    top_p_low,
    top_p_high,
    fit_num=3,
    save_peak_maps=False
):
    tiff_files = sorted(glob.glob(os.path.join(reg_dir, "*.tif")) +
                        glob.glob(os.path.join(reg_dir, "*.tiff")))
    if not tiff_files:
        raise ValueError(f"[X] tif/tiff 없음: {reg_dir}")

    mask_files = sorted([f for f in os.listdir(mask_dir)
                         if f.startswith("roi_mask_slice_") and f.endswith(".npy")])
    if not mask_files:
        raise ValueError(f"[X] roi_mask_slice_XXXX.npy 없음: {mask_dir}")

    peak_dir = os.path.join(mask_dir, "peak_maps") if save_peak_maps else None
    if save_peak_maps:
        os.makedirs(peak_dir, exist_ok=True)

    out_dir = os.path.join(mask_dir, f"filtered_top_{int(round(top_p_low)):02d}_{int(round(top_p_high)):02d}")
    os.makedirs(out_dir, exist_ok=True)

    eff_low = float(low) - float(low_margin)
    eff_high = float(high) + float(high_margin)

    temp_keeps = []
    temp_peak_maps = []
    temp_slice_numbers = []
    temp_shapes = []

    used_slices_for_thr = 0
    first_verbose = True

    # ---------------------------
    # Pass 1) slice별 peak_map, 초기 keep 생성
    # ---------------------------
    for mf in mask_files:
        s1 = parse_slice_number_from_mask_filename(mf)
        if s1 is None:
            continue
        tiff_idx = s1 - 1
        if tiff_idx < 0 or tiff_idx >= len(tiff_files):
            continue

        roi_mask = np.load(os.path.join(mask_dir, mf)).astype(bool)
        if roi_mask.ndim != 2:
            continue

        if roi_mask.sum() == 0:
            keep0 = np.zeros_like(roi_mask, dtype=bool)
            peak_map = np.full(roi_mask.shape, np.nan, dtype=np.float32)
        else:
            img_stack_raw = io.imread(tiff_files[tiff_idx])
            img_stack, eng_aligned = align_energy_axis_for_legacy_fit(
                img_stack_raw, eng,
                energy_axis_reverse=energy_axis_reverse,
                verbose=first_verbose
            )
            first_verbose = False

            raw_peak_map = polynomial_second_fit_separate_raw(img_stack, eng_aligned, fit_num, ev_step)

            target_shape = raw_peak_map.shape
            if roi_mask.shape != target_shape:
                roi_mask = normalize_mask_to_shape(roi_mask, target_shape)

            peak_map = raw_peak_map.copy()
            peak_map[peak_map < eff_low] = np.nan
            peak_map[peak_map > eff_high] = np.nan
            valid_pix = np.isfinite(peak_map)
            peak_map[valid_pix & (peak_map < low)] = float(low)
            peak_map[valid_pix & (peak_map > high)] = float(high)

            if save_peak_maps:
                np.save(os.path.join(peak_dir, f"peak_map_slice_{s1:04d}.npy"), peak_map.astype(np.float32))

            keep0 = roi_mask & np.isfinite(peak_map)

        temp_keeps.append(keep0.astype(bool))
        temp_peak_maps.append(peak_map.astype(np.float32))
        temp_slice_numbers.append(s1)
        temp_shapes.append(keep0.shape)

    if not temp_keeps:
        raise ValueError("[X] 처리 가능한 slice가 없습니다.")

    min_h = min(s[0] for s in temp_shapes)
    min_w = min(s[1] for s in temp_shapes)
    common_shape = (min_h, min_w)

    keep_slices_norm = []
    peak_slices_norm = []
    for keep0, peak_map in zip(temp_keeps, temp_peak_maps):
        if keep0.shape != common_shape:
            keep0 = normalize_mask_to_shape(keep0, common_shape)
        if peak_map.shape != common_shape:
            peak_map = normalize_array_to_shape(peak_map, common_shape, fill_value=np.nan)

        keep_slices_norm.append(keep0)
        peak_slices_norm.append(peak_map)

    # ---------------------------
    # Pass 1.5) 3D singleton 제거 후 threshold 계산용 값 수집
    # ---------------------------
    keep_3d_initial = np.stack(keep_slices_norm, axis=0)
    keep_3d_initial = remove_singleton_voxels_3d(keep_3d_initial, connectivity=1)

    all_vals = []
    for i in range(keep_3d_initial.shape[0]):
        vals = peak_slices_norm[i][keep_3d_initial[i]]
        vals = filter_and_clip_valid_peak_values(vals, low, high, low_margin, high_margin)
        if vals.size:
            all_vals.append(vals.astype(np.float32))
            used_slices_for_thr += 1

    if not all_vals:
        raise ValueError("[X] particle-wide threshold 계산용 ROI peak 모으기 실패 (3D singleton 제거 후 전부 소멸)")

    all_vals = np.concatenate(all_vals, axis=0)
    thr_lo, thr_hi = top_based_thresholds(all_vals, top_p_low, top_p_high)

    if not np.isfinite(thr_lo) or not np.isfinite(thr_hi):
        raise ValueError("[X] threshold 계산 실패(NaN).")

    print(
        f"[Particle-wide] {os.path.basename(reg_dir)}  "
        f"base=[{low:.6f},{high:.6f}]  eff=[{eff_low:.6f},{eff_high:.6f}]  "
        f"TOP[{top_p_low},{top_p_high}] thr=[{thr_lo:.6f},{thr_hi:.6f}]  "
        f"N={all_vals.size}  slices={used_slices_for_thr}"
    )

    # ---------------------------
    # Pass 2) threshold 적용 후 3D singleton 제거
    # ---------------------------
    final_keep_slices = []
    for keep0, peak_map in zip(keep_slices_norm, peak_slices_norm):
        keep = keep0 & np.isfinite(peak_map) & (peak_map >= thr_lo) & (peak_map <= thr_hi)
        final_keep_slices.append(keep.astype(bool))

    final_keep_3d = np.stack(final_keep_slices, axis=0)
    final_keep_3d = remove_singleton_voxels_3d(final_keep_3d, connectivity=1)

    for s1, keep in zip(temp_slice_numbers, final_keep_3d):
        np.save(os.path.join(out_dir, f"filtered_mask_slice_{s1:04d}.npy"), keep.astype(bool))

    meta_txt = os.path.join(out_dir, "threshold_info.txt")
    with open(meta_txt, "w", encoding="utf-8") as f:
        f.write(f"RegisteredFolder: {reg_dir}\n")
        f.write(f"MaskDir: {mask_dir}\n")
        f.write(f"TOP-based range: [{top_p_low},{top_p_high}]\n")
        f.write(f"base_low_eV: {low:.8f}\n")
        f.write(f"base_high_eV: {high:.8f}\n")
        f.write(f"low_margin_eV: {low_margin:.8f}\n")
        f.write(f"high_margin_eV: {high_margin:.8f}\n")
        f.write(f"eff_low_eV: {eff_low:.8f}\n")
        f.write(f"eff_high_eV: {eff_high:.8f}\n")
        f.write(f"thr_lo_eV: {thr_lo:.8f}\n")
        f.write(f"thr_hi_eV: {thr_hi:.8f}\n")
        f.write(f"N_total_valid_roi_voxels_used: {int(all_vals.size)}\n")
        f.write(f"Slices_used: {int(used_slices_for_thr)}\n")
        f.write(f"peak_map_saved_rule: outside eff range -> NaN(discard), margin-inside -> clipped to base range\n")
        f.write(f"singleton_rule: remove isolated 1-voxel connected components in 3D\n")
        f.write(f"singleton_connectivity: face-connected 6-neighbour (connectivity=1)\n")
        f.write(f"common_shape_used: {common_shape[0]},{common_shape[1]}\n")
        f.write(f"fit_num: {fit_num}\n")
        f.write(f"ev_step: {ev_step}\n")
        f.write(f"energy_axis_reverse: {energy_axis_reverse}\n")

    return out_dir, thr_lo, thr_hi, int(all_vals.size), int(used_slices_for_thr)


# =========================================================
# main
# =========================================================
def main():
    Tk().withdraw()

    energy_file = filedialog.askopenfilename(title="energy.txt 선택")
    if not energy_file:
        print("[X] energy.txt 미선택. 종료.")
        return
    eng = np.loadtxt(energy_file)

    ev_step_str = simpledialog.askstring("ev_step", "fit_3D에서 사용한 ev_step(eV)", initialvalue="1")
    try:
        ev_step = float(ev_step_str)
    except Exception:
        ev_step = 1.0

    energy_axis_reverse = messagebox.askyesno(
        "Energy axis reverse?",
        "registered tif를 읽은 뒤 img_stack을 [::-1,:,:]로 뒤집을까요?\n"
        "(기존 fit_3D.batch_fitting과 동일이면 YES 추천)"
    )

    base_range_str = simpledialog.askstring(
        "기준 peak_site 범위 (low,high)",
        "기준 에너지 범위를 입력하세요.\n"
        "예: 8353.6,8356.2",
        initialvalue="8353.6,8356.2"
    )
    try:
        a, b = base_range_str.replace(" ", "").split(",")
        low, high = float(a), float(b)
        if low > high:
            low, high = high, low
    except Exception:
        messagebox.showerror("입력 오류", "기준 범위 형식은 low,high 입니다. 예: 8353.6,8356.2")
        return

    margin_str = simpledialog.askstring(
        "상/하단 마진 (low_margin,high_margin)",
        "유효복셀 범위 = [low-low_margin, high+high_margin]\n"
        "예: 0,0   또는   0.5,0.5",
        initialvalue="0,0"
    )
    try:
        lm, hm = margin_str.replace(" ", "").split(",")
        low_margin, high_margin = float(lm), float(hm)
        if low_margin < 0:
            low_margin = 0.0
        if high_margin < 0:
            high_margin = 0.0
    except Exception:
        messagebox.showerror("입력 오류", "마진 형식은 low_margin,high_margin 입니다. 예: 0.5,0.5")
        return

    phase_name = ask_phase_selection()
    phase_excel = ask_phase_fraction_excel()
    frac_df = load_phase_fraction_table(phase_excel)

    soc0_energy, soc100_energy = ask_soc_reference_energies()

    save_peak_maps = messagebox.askyesno(
        "peak_map 저장?",
        "각 registered 입자별로 peak_map을 mask 폴더에 peak_maps/로 저장할까요?\n"
        "YES면 다음부터 같은 입자에서 더 빠르게 filtered mask를 만들 수 있습니다."
    )

    fit_num = 3

    eff_low = low - low_margin
    eff_high = high + high_margin
    print(f"✅ base range = [{low:.6f}, {high:.6f}]")
    print(f"✅ margins    = [{low_margin:.6f}, {high_margin:.6f}]")
    print(f"✅ valid range= [{eff_low:.6f}, {eff_high:.6f}]")
    print(f"✅ phase mask = {phase_name}")
    print(f"✅ phase table= {phase_excel}")
    print(f"✅ SOC ref    = {soc0_energy:.6f} eV -> SOC 0, {soc100_energy:.6f} eV -> SOC 100")

    parents = ask_parent_directories()
    if not parents:
        print("[X] 상위 폴더 미선택. 종료.")
        return

    total_done = 0

    for parent in parents:
        entries = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        star_folders = [d for d in entries if "★" in d]

        if not star_folders:
            print(f"[SKIP] '★' 폴더가 없습니다: {parent}")
            continue

        print(f"\n#############################")
        print(f"상위 폴더 처리 중: {parent}")
        print(f"#############################")

        for star_name in sorted(star_folders):
            n = extract_registered_number_from_star_folder(star_name)
            if n is None:
                print(f"[SKIP] 번호 추출 실패: {star_name}")
                continue

            reg_dir = os.path.join(parent, f"registered{n}")
            mask_dir = os.path.join(reg_dir, f"{n}")

            if not os.path.isdir(reg_dir):
                print(f"[SKIP] registered 없음: {reg_dir}   (from {star_name})")
                continue
            if not os.path.isdir(mask_dir):
                print(f"[SKIP] mask_dir 없음: {mask_dir}   (from {star_name})")
                continue

            try:
                soc_value, soc_source, mean_peak_eV = get_soc_for_registered(reg_dir, soc0_energy, soc100_energy)
                soc_used, h2_frac, h3_frac = interpolate_h2_h3_from_soc(soc_value, frac_df)
                top_p_low, top_p_high = percentile_range_from_phase_fraction(phase_name, h2_frac, h3_frac)

                print(
                    f"[SOC] {star_name} -> registered{n}  "
                    f"mean_peak={mean_peak_eV:.4f} eV  "
                    f"SOC={soc_value:.4f} (source={soc_source})  "
                    f"interp_SOC={soc_used:.4f}  H2={h2_frac:.4f}  H3={h3_frac:.4f}  "
                    f"phase={phase_name}  top=[{top_p_low:.2f},{top_p_high:.2f}]"
                )

                out_dir, thr_lo, thr_hi, Nvox, Nsli = process_registered_make_filtered_masks(
                    reg_dir=reg_dir,
                    mask_dir=mask_dir,
                    eng=eng,
                    ev_step=ev_step,
                    energy_axis_reverse=energy_axis_reverse,
                    low=low,
                    high=high,
                    low_margin=low_margin,
                    high_margin=high_margin,
                    top_p_low=top_p_low,
                    top_p_high=top_p_high,
                    fit_num=fit_num,
                    save_peak_maps=save_peak_maps
                )

                meta_append = os.path.join(out_dir, "threshold_info.txt")
                with open(meta_append, "a", encoding="utf-8") as f:
                    f.write(f"phase_selected: {phase_name}\n")
                    f.write(f"particle_mean_peak_eV: {mean_peak_eV:.8f}\n")
                    f.write(f"soc_detected: {soc_value:.8f}\n")
                    f.write(f"soc_source: {soc_source}\n")
                    f.write(f"soc_used_for_interp: {soc_used:.8f}\n")
                    f.write(f"soc0_energy_ref: {soc0_energy:.8f}\n")
                    f.write(f"soc100_energy_ref: {soc100_energy:.8f}\n")
                    f.write(f"h2_fraction_interp: {h2_frac:.8f}\n")
                    f.write(f"h3_fraction_interp: {h3_frac:.8f}\n")
                    f.write(f"phase_fraction_excel: {phase_excel}\n")

                print(
                    f"[OK] {star_name} -> registered{n}  out={out_dir}  "
                    f"thr=[{thr_lo:.4f},{thr_hi:.4f}]  N={Nvox}  slices={Nsli}"
                )
                total_done += 1

            except Exception as e:
                messagebox.showerror("처리 실패", f"{star_name}\n{reg_dir}\n{mask_dir}\n\n{e}")

    messagebox.showinfo("완료", f"filtered mask 생성 완료!\n처리된 registered 개수: {total_done}")


if __name__ == "__main__":
    main()
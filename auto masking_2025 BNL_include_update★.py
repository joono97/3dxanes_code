#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from skimage import io
from tkinter import Tk, filedialog, simpledialog, messagebox
from numpy.polynomial import polynomial as poly
from skimage.measure import find_contours
from skimage.filters import threshold_otsu
import pandas as pd


# =========================


# ROI 선택 (M 토글: 수동만 저장 / 기본: 자동ROI ∪와 교집합)
#  + Space: 이전 슬라이스ㅜ
#  + B/b: 현재 포함 이후 전체 블랙마스크




# =========================
def get_polygon_mask_click_to_select(img_2d, slice_idx=None):
    norm_img = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)
    thresh = threshold_otsu(norm_img) * 1.1

    binary = norm_img > thresh

    contours = find_contours(binary, level=0.5)
    fig, ax = plt.subplots()
    ax.imshow(norm_img, cmap='gray')

    H, W = img_2d.shape
    xg, yg = np.meshgrid(np.arange(W), np.arange(H))


    coords = np.vstack((xg.ravel(), yg.ravel())).T

    auto_paths = []
    from matplotlib.path import Path as _MplPath
    if contours:
        for contour in contours:
            verts = contour[:, ::-1]
            codes = np.full(len(verts) + 1, _MplPath.LINETO, dtype=np.uint8)
            codes[0] = _MplPath.MOVETO
            codes[-1] = _MplPath.CLOSEPOLY
            verts_closed = np.vstack([verts, verts[0]])
            path = _MplPath(verts_closed, codes)
            auto_paths.append(path)
            ax.plot(verts[:, 0], verts[:, 1], linewidth=1.2, color='red')




    union_auto_mask = np.zeros_like(img_2d, dtype=bool)
    for path in auto_paths:
        union_auto_mask |= path.contains_points(coords).reshape(H, W)

    def title_text(manual_only):
        base = (f"[ROI 자동 후보] Slice {slice_idx+1} – "
                f"우클릭: 자동 선택 / 좌클릭: 수동 / "
                f"Enter: 확정 / Space: 이전 / B: 이후 전부 블랙")
        return base + ("   |   Mode: M=수동만 저장"
                       if manual_only else "   |   Mode: ∩=자동과 교집합")

    ax.set_title(title_text(False))
    mode_label = ax.text(
        0.02, 0.02, "Mode: ∩(교집합)", color='yellow',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor='black', alpha=0.4, pad=3)
    )

    selected_mask = None
    manual_points = []
    polygon_patch = None
    go_prev = False
    manual_only = False
    black_rest = False

    def redraw_auto_contours():
        if contours:
            for contour in contours:
                verts = contour[:, ::-1]
                ax.plot(verts[:, 0], verts[:, 1], linewidth=1.2, color='red')

    def update_polygon():
        nonlocal polygon_patch
        if polygon_patch:
            polygon_patch.remove()
        if len(manual_points) >= 3:
            polygon_patch = Polygon(
                manual_points, closed=True, fill=True,
                edgecolor='lime', facecolor='lime', alpha=0.3
            )
            ax.add_patch(polygon_patch)
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal selected_mask, manual_points
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 3:
            if not auto_paths:
                print("[!] 자동 ROI가 없어 우클릭 선택 불가 (M을 눌러 수동만 저장 가능)")
                return
            x_click, y_click = event.xdata, event.ydata
            for path in auto_paths:
                if path.contains_point((x_click, y_click)):
                    print("[✔] 우클릭: 해당 자동 ROI 선택됨")
                    selected_mask = path.contains_points(coords).reshape(H, W)
                    plt.close(fig)
                    return
            print("[!] 우클릭 위치에 해당하는 ROI가 없음")

        elif event.button == 1:
            manual_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'o', color='lime', markersize=4)
            update_polygon()

    def on_key(event):
        nonlocal selected_mask, go_prev, manual_only, mode_label, black_rest
        if event.key in ('m', 'M'):
            manual_only = not manual_only
            mode_label.set_text("Mode: M(수동만)" if manual_only else "Mode: ∩(교집합)")
            ax.set_title(title_text(manual_only))
            fig.canvas.draw_idle()

        elif event.key == 'enter':
            if manual_points and len(manual_points) >= 3:
                print("[✔] 수동 다각형 ROI 닫힘")
                poly_path = MplPath(manual_points, closed=True)
                manual_mask = poly_path.contains_points(coords).reshape(H, W)

                if manual_only:
                    selected_mask = manual_mask
                    if not np.any(selected_mask):
                        print("[X] 수동 ROI가 비어 있음 → 빈 마스크 저장")
                else:
                    intersect_mask = manual_mask & union_auto_mask
                    if np.any(intersect_mask):
                        selected_mask = intersect_mask
                        print("[✓] 교집합 존재 → 교집합을 최종 ROI로 사용")
                    else:
                        selected_mask = np.zeros_like(img_2d, dtype=bool)
                        print("[X] 교집합 없음 → 최종 ROI는 빈 마스크(교집합만 허용)")
            else:
                selected_mask = np.zeros_like(img_2d, dtype=bool)
                print("[⚫] ROI를 지정하지 않음 → 빈 마스크 생성")
            plt.close(fig)

        elif event.key == 'backspace':
            if manual_points:
                manual_points.pop()
                ax.cla()
                ax.imshow(norm_img, cmap='gray')
                redraw_auto_contours()
                for p in manual_points:
                    ax.plot(p[0], p[1], 'o', color='lime', markersize=4)
                update_polygon()
                ax.set_title(title_text(manual_only))
                mode_label = ax.text(
                    0.02, 0.02,
                    "Mode: M(수동만)" if manual_only else "Mode: ∩(교집합)",
                    color='yellow', transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.4, pad=3)
                )
                fig.canvas.draw_idle()

        elif event.key in (' ', 'space'):
            go_prev = True
            print("[↩] 스페이스: 이전 슬라이스로 돌아가기")
            plt.close(fig)

        elif event.key in ('b', 'B'):
            black_rest = True
            print("[■] B키: 현재 슬라이스 포함 이후 모든 선택슬라이스를 블랙 마스크로 설정")
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if go_prev:
        return None, 'prev'
    if black_rest:
        return None, 'black_rest'
    if selected_mask is None:
        print("[X] ROI가 선택되지 않음 – 슬라이스 건너뜀")
    return selected_mask, 'ok'


# =========================================================
# energy 축 정합
# =========================================================
def align_energy_axis_for_legacy_fit(img_stack, eng, energy_axis_reverse=True, verbose=True):
    eng = np.array(eng, dtype=float).ravel()

    if img_stack.ndim != 3:
        raise ValueError(f"[X] img_stack must be 3D (E,H,W). Got shape={img_stack.shape}")

    if img_stack.shape[0] != len(eng):
        raise ValueError(
            f"[X] energy length mismatch: img_stack E={img_stack.shape[0]} vs len(eng)={len(eng)}"
        )

    if energy_axis_reverse:
        img_stack = img_stack[::-1, :, :]
        if verbose:
            print("[i] img_stack energy axis reversed: img_stack = img_stack[::-1,:,:]")

    if eng[0] > eng[-1]:
        eng = eng[::-1]
        if verbose:
            print("[i] eng reversed to increasing order")

    diffs = np.diff(eng)
    if np.any(diffs <= 0):
        raise ValueError(
            "[X] eng가 단조 증가가 아닙니다. energy.txt 순서가 섞였거나, "
            "stack 순서와 energy.txt가 매칭되지 않을 수 있습니다."
        )

    return img_stack, eng


# =========================
# 피크맵 (국소 2차 피팅)
# =========================
def polynomial_second_fit_separate(img_stack, eng, fit_num, ev_step, peakref=None):
    y = img_stack.reshape(len(img_stack), -1)
    y_new = np.zeros((fit_num * 2 + 1, y.shape[1]))
    mp_list = []

    for i in range(y.shape[1]):
        a = np.argmax(y[:, i])
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

    with np.errstate(divide='ignore', invalid='ignore'):
        x0 = -(coefs[1] / 2 / coefs[2]) + eng[ind]

    x0[~np.isfinite(x0)] = np.nan
    peak_map = x0.reshape(img_stack.shape[1], img_stack.shape[2])
    return peak_map


def save_black_mask(mask_dir, idx, shape_hw):
    path = os.path.join(mask_dir, f"roi_mask_slice_{idx + 1:04d}.npy")
    np.save(path, np.zeros(shape_hw, dtype=bool))
    print(f"[↷] 이 슬라이스 스킵/블랙 → 저장: {path}")


# =========================
# 유효픽셀 먼저 판정하고, 유효픽셀만 clip해서 평균
# =========================
def compute_stats_single_criterion(roi_values, low, high, low_margin, high_margin):
    roi_values = np.asarray(roi_values, dtype=float)
    roi_values = roi_values[np.isfinite(roi_values)]
    total = int(roi_values.size)

    eff_low = float(low) - float(low_margin)
    eff_high = float(high) + float(high_margin)

    valid_mask = (roi_values >= eff_low) & (roi_values <= eff_high)
    valid_pixels = int(np.count_nonzero(valid_mask))
    ratio = (100.0 * valid_pixels / total) if total else 0.0

    if valid_pixels > 0:
        v = roi_values[valid_mask].astype(float, copy=False)
        v_clipped = np.clip(v, float(low), float(high))
        roi_mean = float(np.mean(v_clipped))
    else:
        v_clipped = np.array([], dtype=float)
        roi_mean = np.nan

    return v_clipped, valid_pixels, total, ratio, roi_mean, eff_low, eff_high


# =========================
# registeredX 탐색
# =========================
def find_registered_folders(parent_dir):
    items = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.fullmatch(r"registered(\d+)", name)
        if m:
            items.append((int(m.group(1)), full))
    return sorted(items, key=lambda x: x[0])


# =========================
# 여러 상위폴더 선택
# =========================
def ask_parent_directories():
    parents = []
    while True:
        folder = filedialog.askdirectory(
            title="registeredX 폴더들이 모여있는 상위 폴더 선택 (취소 누르면 선택 종료)"
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


# =========================
# 혼합 지정(단일/범위) 지원: 1,11~13 형태
# =========================
def ask_slice_range(N):
    default = f"1~{N}"
    s = simpledialog.askstring(
        "처리 슬라이스 범위",
        f"총 {N}개 슬라이스.\n콤마(,)로 구분하여 단일/범위를 섞어 입력하세요.\n"
        f"예: 1,11~13  또는  3-5,7  (기본: {default})",
        initialvalue=default
    )
    if not s:
        s = default


    tokens = [t.strip() for t in s.replace(' ', '').split(',') if t.strip()]
    selected = set()

    def clamp_int(x):
        try:
            v = int(x)
        except Exception:
            return None
        return max(1, min(N, v))

    for tok in tokens:
        if '~' in tok or '-' in tok:
            sep = '~' if '~' in tok else '-'
            a, b = tok.split(sep, 1)
            a = clamp_int(a)
            b = clamp_int(b)
            if a is None or b is None:
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            for k in range(lo, hi + 1):
                selected.add(k - 1)
        else:
            v = clamp_int(tok)
            if v is not None:
                selected.add(v - 1)

    if not selected:
        return list(range(N))
    return sorted(selected)


# =========================
# 기존 CSV 로드
# =========================
def load_existing_csv(csv_path):
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"[!] 기존 CSV 로드 실패: {e} → 새로 생성")
    cols = ['Slice', 'Mean Peak (eV)', 'Valid Pixels', 'Total ROI Pixels',
            'Valid %', 'Valid Range Low (eV)', 'Valid Range High (eV)',
            'Particle Mean Peak (eV)']
    return pd.DataFrame(columns=cols)


# =========================
# 현재 저장된 전체 마스크 기준으로
# 전체 유효복셀을 다시 모아서 Particle Mean Peak 계산
# =========================
def recompute_particle_mean_from_all_slices(
    tiff_files,
    mask_dir,
    eng,
    energy_axis_reverse,
    fit_num,
    ev_step,

    peakref,
    default_valid_range,
    low_margin,
    high_margin
):
    all_valid_voxels = []

    for i, tiff_path in enumerate(tiff_files):
        slice_1based = i + 1
        mask_path = os.path.join(mask_dir, f"roi_mask_slice_{slice_1based:04d}.npy")
        if not os.path.exists(mask_path):
            continue

        try:
            mask = np.load(mask_path).astype(bool)
        except Exception as e:
            print(f"[SKIP] mask load 실패: {mask_path} / {e}")
            continue

        if not np.any(mask):
            continue

        try:
            img_stack_raw = io.imread(tiff_path)
            img_stack, eng_aligned = align_energy_axis_for_legacy_fit(
                img_stack_raw, eng,
                energy_axis_reverse=energy_axis_reverse,
                verbose=False
            )
            peak_map = polynomial_second_fit_separate(
                img_stack, eng_aligned, fit_num, ev_step, peakref
            )
        except Exception as e:
            print(f"[SKIP] peak fit 실패: {tiff_path} / {e}")
            continue

        if mask.shape != peak_map.shape:
            print(f"[SKIP] shape mismatch: slice {slice_1based} mask={mask.shape} peak={peak_map.shape}")
            continue

        roi_values = peak_map[mask]
        use_low, use_high = default_valid_range
        v_clipped, valid_pixels, _, _, _, _, _ = compute_stats_single_criterion(
            roi_values, use_low, use_high, low_margin, high_margin
        )

        if valid_pixels > 0 and v_clipped.size > 0:
            all_valid_voxels.append(v_clipped)

    if len(all_valid_voxels) == 0:
        return np.nan

    all_valid_voxels = np.concatenate(all_valid_voxels, axis=0)
    return float(np.mean(all_valid_voxels))


# =========================
# 병합 + 전체 유효복셀 직접 평균 반영
# =========================
def merge_updates_and_recompute(old_df, updates, particle_mean_peak):
    if old_df is None or old_df.empty:
        cols = ['Slice', 'Mean Peak (eV)', 'Valid Pixels', 'Total ROI Pixels',
                'Valid %', 'Valid Range Low (eV)', 'Valid Range High (eV)',
                'Particle Mean Peak (eV)']
        old_df = pd.DataFrame(columns=cols)

    df = old_df[old_df['Slice'].astype(str).str.lower() != 'average'].copy()
    df['Slice_num'] = pd.to_numeric(df['Slice'], errors='coerce')
    df = df[df['Slice_num'].notna()].copy()
    df['Slice'] = df['Slice_num'].astype(int)
    df.drop(columns=['Slice_num'], inplace=True)

    upd_df = pd.DataFrame(updates).copy()
    upd_df['Slice'] = pd.to_numeric(upd_df['Slice'], errors='coerce').astype('Int64')
    upd_df = upd_df[upd_df['Slice'].notna()].copy()
    upd_df['Slice'] = upd_df['Slice'].astype(int)

    if not upd_df.empty:
        df = df[~df['Slice'].isin(set(upd_df['Slice']))]

    merged = pd.concat([df, upd_df], ignore_index=True)
    merged.sort_values(by=['Slice'], kind='mergesort', inplace=True)
    merged = merged.drop_duplicates(subset=['Slice'], keep='last')

    for c in ['Valid Pixels', 'Total ROI Pixels', 'Mean Peak (eV)',
              'Valid Range Low (eV)', 'Valid Range High (eV)', 'Valid %',
              'Particle Mean Peak (eV)']:
        if c not in merged.columns:
            merged[c] = np.nan

    v = pd.to_numeric(merged['Valid Pixels'], errors='coerce')
    t = pd.to_numeric(merged['Total ROI Pixels'], errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        merged['Valid %'] = (100.0 * v / t).round(2)
    merged['Valid %'] = merged['Valid %'].fillna(0.0)

    merged['Particle Mean Peak (eV)'] = (
        round(particle_mean_peak, 3) if np.isfinite(particle_mean_peak) else np.nan
    )

    avg_row = pd.DataFrame([{
        'Slice': 'average',
        'Mean Peak (eV)': f"{particle_mean_peak:.3f}" if np.isfinite(particle_mean_peak) else "",
        'Valid Pixels': "",
        'Total ROI Pixels': "",
        'Valid %': "",
        'Valid Range Low (eV)': "",
        'Valid Range High (eV)': "",
        'Particle Mean Peak (eV)': ""
    }])

    cols = ['Slice', 'Mean Peak (eV)', 'Valid Pixels', 'Total ROI Pixels',
            'Valid %', 'Valid Range Low (eV)', 'Valid Range High (eV)',
            'Particle Mean Peak (eV)']
    merged = merged[cols]
    final_df = pd.concat([merged, avg_row], ignore_index=True)
    return final_df


# =========================
# 공통 파라미터 입력
# =========================
def ask_common_parameters():
    energy_file = filedialog.askopenfilename(title="[File] energy.txt 파일 선택")
    if not energy_file:
        raise ValueError("❌ energy.txt 미선택")

    eng = np.loadtxt(energy_file)

    # 기본값을 먼저 지정해서, 어떤 분기를 타더라도
    # local variable 'energy_axis_reverse' referenced before assignment 에러를 방지
    energy_axis_reverse = False

    ev_step_str = simpledialog.askstring(
        "ev_step 입력",
        "피크피팅에 사용할 ev_step(eV)을 입력하세요.\n"
        "※ fit_3D.main에서 준 ev_step과 동일해야 합니다. (예: 1)",
        initialvalue="1"
    )
    try:

        ev_step = float(ev_step_str)
    except Exception:
        ev_step = 0.5


    energy_axis_reverse = messagebox.askyesno(
        "Energy axis reverse?",
        "registered tif를 불러온 뒤 img_stack을 [::-1,:,:]로 뒤집을까요?\n\n"
        "- fit_3D.batch_fitting은 기본적으로 뒤집습니다.\n"
        "- 예(YES): 뒤집음(추천)\n"
        "- 아니오(NO): 뒤집지 않음"
    )

    base_range_str = simpledialog.askstring(
        "기준 peak_site 범위 (low,high)",
        "최종 peak_site hard clip 및 평균 계산 기준 범위를 입력하세요.\n"
        "예: 8340,8360  또는  8353.6,8356.2",
        initialvalue="8340,8360"
    )
    try:
        low_s, high_s = base_range_str.replace(' ', '').split(',')
        base_low = float(low_s)
        base_high = float(high_s)
        if base_low > base_high:
            base_low, base_high = base_high, base_low
    except Exception:
        raise ValueError("❌ 기준 peak_site 범위 입력 형식 오류")

    margin_str = simpledialog.askstring(
        "상/하단 마진 (eV)",
        "유효픽셀 범위 = [low-low_margin, high+high_margin]\n"
        "형식: low_margin,high_margin\n"
        "예: 0,0   또는   0.5,0.5",
        initialvalue="0,0"
    )
    try:
        lm_s, hm_s = margin_str.replace(' ', '').split(',')
        low_margin = float(lm_s)
        high_margin = float(hm_s)
        if low_margin < 0:
            low_margin = 0.0
        if high_margin < 0:
            high_margin = 0.0
    except Exception:
        low_margin, high_margin = 0.0, 0.0

    min_valid_percent_str = simpledialog.askstring(
        "최소 유효 비율(%)",
        "슬라이스 자동 통과를 위한 최소 유효 비율(%)을 입력하세요:",
        initialvalue="70"
    )
    try:
        min_valid_percent = float(min_valid_percent_str)
    except Exception:
        min_valid_percent = 70.0

    fit_num = 3
    peakref = [base_low, base_high]
    default_valid_range = [base_low, base_high]

    print(f"✅ base range         = [{peakref[0]:.4f}, {peakref[1]:.4f}]")
    print(f"✅ valid range        = [{default_valid_range[0]:.4f}, {default_valid_range[1]:.4f}]")
    print(f"✅ margins            = low={low_margin:.4f}, high={high_margin:.4f}")
    print(f"✅ effective valid    = [{base_low-low_margin:.4f}, {base_high+high_margin:.4f}]")
    print("✅ voxel peak_site는 raw로 계산 후, 유효픽셀 판정 뒤에만 clip 적용")

    return {
        "energy_file": energy_file,
        "eng": eng,
        "ev_step": ev_step,
        "energy_axis_reverse": energy_axis_reverse,
        "base_low": base_low,
        "base_high": base_high,
        "low_margin": low_margin,
        "high_margin": high_margin,
        "min_valid_percent": min_valid_percent,
        "fit_num": fit_num,
        "peakref": peakref,
        "default_valid_range": default_valid_range
    }


# =========================
# interactive 모드
# =========================
def main_interactive():
    Tk().withdraw()

    tiff_dir = filedialog.askdirectory(title="[Slice] TIFF 슬라이스 폴더 선택")
    if not tiff_dir:
        print("[X] TIFF 폴더 미선택")
        return

    params = ask_common_parameters()

    mask_subfolder_name = simpledialog.askstring("ROI 폴더 이름", "ROI 마스크를 저장할 하위 폴더 이름을 입력하세요:")
    if not mask_subfolder_name:
        print("[X] ROI 마스크 폴더 이름이 지정되지 않았습니다. 종료합니다.")
        return

    mask_dir = os.path.join(tiff_dir, mask_subfolder_name)
    os.makedirs(mask_dir, exist_ok=True)

    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "*.tif")) + glob.glob(os.path.join(tiff_dir, "*.tiff")))
    if len(tiff_files) == 0:
        print("[X] TIFF 슬라이스 없음")
        return

    sel_list = ask_slice_range(len(tiff_files))
    if len(sel_list) == 0:
        print("[X] 선택된 슬라이스 없음")
        return

    csv_path = os.path.join(tiff_dir, "roi_peak_summary.csv")
    old_df = load_existing_csv(csv_path)
    roi_updates = []

    idx_ptr = 0
    black_rest_mode = False

    while idx_ptr < len(sel_list):
        i = sel_list[idx_ptr]
        slice_1based = i + 1
        tiff_path = tiff_files[i]

        img_stack_raw = io.imread(tiff_path)
        try:
            img_stack, eng_aligned = align_energy_axis_for_legacy_fit(
                img_stack_raw,
                params["eng"],
                energy_axis_reverse=params["energy_axis_reverse"],
                verbose=(idx_ptr == 0)
            )
        except Exception as e:
            print(f"[SKIP] energy 정합 실패: {tiff_path} / {e}")
            idx_ptr += 1
            continue

        if black_rest_mode:
            save_black_mask(mask_dir, i, img_stack.shape[1:])
            roi_updates.append({
                'Slice': slice_1based,
                'Mean Peak (eV)': np.nan,
                'Valid Pixels': 0,
                'Total ROI Pixels': 0,
                'Valid %': 0.0,
                'Valid Range Low (eV)': float(params["default_valid_range"][0]),
                'Valid Range High (eV)': float(params["default_valid_range"][1]),
                'Particle Mean Peak (eV)': np.nan,
            })
            idx_ptr += 1
            continue

        display_img = np.nanmean(img_stack, axis=0)
        selected_mask, action = get_polygon_mask_click_to_select(display_img, slice_idx=i)

        if action == 'prev':
            idx_ptr = max(0, idx_ptr - 1)
            continue

        if action == 'black_rest':
            black_rest_mode = True
            save_black_mask(mask_dir, i, img_stack.shape[1:])
            roi_updates.append({
                'Slice': slice_1based,
                'Mean Peak (eV)': np.nan,
                'Valid Pixels': 0,
                'Total ROI Pixels': 0,
                'Valid %': 0.0,
                'Valid Range Low (eV)': float(params["default_valid_range"][0]),
                'Valid Range High (eV)': float(params["default_valid_range"][1]),
                'Particle Mean Peak (eV)': np.nan,
            })
            idx_ptr += 1
            continue

        if selected_mask is None:
            print(f"[SKIP] ROI 선택 안 됨: slice {slice_1based}")
            idx_ptr += 1
            continue

        mask_path = os.path.join(mask_dir, f"roi_mask_slice_{slice_1based:04d}.npy")
        np.save(mask_path, selected_mask.astype(bool))
        print(f"[✔] ROI mask 저장: {mask_path}")

        peak_map = polynomial_second_fit_separate(
            img_stack, eng_aligned, params["fit_num"], params["ev_step"], params["peakref"]
        )

        if selected_mask.shape != peak_map.shape:
            print(f"[SKIP] shape mismatch: slice {slice_1based} mask={selected_mask.shape} peak={peak_map.shape}")
            idx_ptr += 1
            continue

        roi_values = peak_map[selected_mask]
        use_low, use_high = params["default_valid_range"]

        v_clipped, valid_pixels, total_roi_pixels, ratio, roi_mean, eff_low, eff_high = compute_stats_single_criterion(
            roi_values, use_low, use_high, params["low_margin"], params["high_margin"]
        )

        if ratio >= params["min_valid_percent"]:
            print(f"[AUTO] slice {slice_1based}: valid={ratio:.2f}% ≥ {params['min_valid_percent']:.2f}%")
        else:
            print(f"[WARN] slice {slice_1based}: valid={ratio:.2f}% < {params['min_valid_percent']:.2f}%")

        roi_updates.append({
            'Slice': slice_1based,
            'Mean Peak (eV)': float(np.round(roi_mean, 3)) if np.isfinite(roi_mean) else np.nan,
            'Valid Pixels': int(valid_pixels),
            'Total ROI Pixels': int(total_roi_pixels),
            'Valid %': float(np.round(ratio, 2)),
            'Valid Range Low (eV)': float(use_low),
            'Valid Range High (eV)': float(use_high),
            'Particle Mean Peak (eV)': np.nan,
        })

        idx_ptr += 1

    if len(roi_updates) == 0:


        print("[X] 업데이트할 ROI 없음")
        return
    particle_mean_peak = recompute_particle_mean_from_all_slices(
        tiff_files=tiff_files,
        mask_dir=mask_dir,
        eng=params["eng"],
        energy_axis_reverse=params["energy_axis_reverse"],
        fit_num=params["fit_num"],
        ev_step=params["ev_step"],
        peakref=params["peakref"],
        default_valid_range=params["default_valid_range"],
        low_margin=params["low_margin"],
        high_margin=params["high_margin"]
    )

    new_df = merge_updates_and_recompute(old_df, roi_updates, particle_mean_peak)
    new_df.to_csv(csv_path, index=False)
    print(f"[✔] CSV 저장 완료: {csv_path}")
    print(f"[✔] Particle Mean Peak (전체 유효복셀 직접 평균): {particle_mean_peak:.6f}" if np.isfinite(particle_mean_peak) else
          "[WARN] 전체 유효복셀 없음 → Particle Mean Peak = NaN")


# =========================
# batch CSV only
# =========================
def update_csv_for_one_registered(tiff_dir, mask_dir, params):
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "*.tif")) + glob.glob(os.path.join(tiff_dir, "*.tiff")))
    if len(tiff_files) == 0:
        print(f"[SKIP] TIFF 없음: {tiff_dir}")
        return False

    csv_path = os.path.join(tiff_dir, "roi_peak_summary.csv")
    old_df = load_existing_csv(csv_path)
    roi_updates = []

    for i, tiff_path in enumerate(tiff_files):
        slice_1based = i + 1
        mask_path = os.path.join(mask_dir, f"roi_mask_slice_{slice_1based:04d}.npy")

        if not os.path.exists(mask_path):
            continue

        try:
            mask = np.load(mask_path).astype(bool)
        except Exception as e:
            print(f"[SKIP] mask load 실패: {mask_path} / {e}")
            continue

        img_stack_raw = io.imread(tiff_path)
        try:
            img_stack, eng_aligned = align_energy_axis_for_legacy_fit(
                img_stack_raw,
                params["eng"],
                energy_axis_reverse=params["energy_axis_reverse"],
                verbose=False
            )
        except Exception as e:
            print(f"[SKIP] energy 정합 실패: {tiff_path} / {e}")
            continue

        peak_map = polynomial_second_fit_separate(
            img_stack, eng_aligned, params["fit_num"], params["ev_step"], params["peakref"]
        )




        if mask.shape != peak_map.shape:
            print(f"[SKIP] shape mismatch: slice {slice_1based} mask={mask.shape} peak={peak_map.shape}")
            continue

        roi_values = peak_map[mask]
        use_low, use_high = params["default_valid_range"]

        v_clipped, valid_pixels, total_roi_pixels, ratio, roi_mean, eff_low, eff_high = compute_stats_single_criterion(
            roi_values, use_low, use_high, params["low_margin"], params["high_margin"]
        )

        roi_updates.append({
            'Slice': slice_1based,
            'Mean Peak (eV)': float(np.round(roi_mean, 3)) if np.isfinite(roi_mean) else np.nan,
            'Valid Pixels': int(valid_pixels),
            'Total ROI Pixels': int(total_roi_pixels),
            'Valid %': float(np.round(ratio, 2)),
            'Valid Range Low (eV)': float(use_low),
            'Valid Range High (eV)': float(use_high),
            'Particle Mean Peak (eV)': np.nan,
        })

    if len(roi_updates) == 0:
        print(f"[SKIP] 업데이트할 slice 없음: {tiff_dir}")
        return False

    particle_mean_peak = recompute_particle_mean_from_all_slices(
        tiff_files=tiff_files,
        mask_dir=mask_dir,
        eng=params["eng"],
        energy_axis_reverse=params["energy_axis_reverse"],
        fit_num=params["fit_num"],
        ev_step=params["ev_step"],
        peakref=params["peakref"],
        default_valid_range=params["default_valid_range"],
        low_margin=params["low_margin"],
        high_margin=params["high_margin"]
    )

    new_df = merge_updates_and_recompute(old_df, roi_updates, particle_mean_peak)
    new_df.to_csv(csv_path, index=False)
    print(f"[✔] CSV 업데이트 완료: {csv_path}")
    print(f"[✔] Particle Mean Peak (전체 유효복셀 직접 평균): {particle_mean_peak:.6f}" if np.isfinite(particle_mean_peak) else
          "[WARN] 전체 유효복셀 없음 → Particle Mean Peak = NaN")
    return True


def main_batch_update_csv_only():
    Tk().withdraw()

    params = ask_common_parameters()

    parents = ask_parent_directories()
    if not parents:
        print("[X] 상위 폴더 미선택")
        return

    done = 0
    skipped = 0

    for parent in parents:
        reg_list = find_registered_folders(parent)
        


        if len(reg_list) == 0:
            print(f"[X] registeredX 폴더가 없습니다: {parent}")
            skipped += 1
            continue

        print(f"✅ Found {len(reg_list)} registered folders in: {parent}")

        for n, reg_dir in reg_list:
            mask_dir = os.path.join(reg_dir, f"{n}")
            if not os.path.isdir(mask_dir):
                print(f"[SKIP] 마스크 폴더 없음: {mask_dir}")

                skipped += 1
                continue

            ok = update_csv_for_one_registered(reg_dir, mask_dir, params)
            if ok:
                done += 1
            else:
                skipped += 1

    print("\n=========================")
    print(f"완료: {done}")
    print(f"스킵: {skipped}")
    print("=========================")


# =========================
# 실행 메뉴
# =========================
def main():
    root = Tk()
    root.withdraw()

    use_interactive = messagebox.askyesno(
        "실행 모드 선택",
        "YES: 인터랙티브 ROI 선택 + CSV 갱신\n"
        "NO : 기존 ROI mask를 이용한 batch CSV only 갱신"
    )

    if use_interactive:
        main_interactive()
    else:
        main_batch_update_csv_only()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            root = Tk()


            root.withdraw()
            messagebox.showerror("오류", str(e))
            root.destroy()
        except Exception:
            pass
        print("오류:", e)
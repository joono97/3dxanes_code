import os
import numpy as np
import tifffile
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tifffile import imsave

def get_tiff_files(directory):
    files = []
    for f in os.listdir(directory):
        fl = f.lower()
        if fl.startswith("._"):  # ✅ AppleDouble 파일 제외
            continue
        if fl.endswith((".tif", ".tiff")):
            files.append(os.path.join(directory, f))
    return sorted(files)

def load_and_crop_volume(directory, roi_x, roi_y, roi_width, roi_height, slice_range_start, slice_range_end):
    """
    Load a 3D volume slice-by-slice from TIFF files in the given directory,
    crop each slice using the fixed ROI (x,y only), and stack them into a 3D NumPy array.
    Then, crop the volume in the z-dimension using slice_range_start and slice_range_end.

    TIFF가 없으면 에러로 멈추지 않고 None을 반환해서 상위 loop에서 skip하게 한다.
    """
    file_names = get_tiff_files(directory)
    if not file_names:
        print(f"⚠️ No TIFF files found. Skip this folder: {directory}")
        return None

    cropped_slices = []
    for f in file_names:
        try:
            slice_img = tifffile.imread(f)
            if slice_img.ndim != 2:
                raise ValueError(f"Expected 2D image, got shape {slice_img.shape} in file {f}")
            # Crop using numpy slicing (round ROI coordinates to integers)
            x0 = int(round(roi_x))
            y0 = int(round(roi_y))
            x1 = x0 + int(round(roi_width))
            y1 = y0 + int(round(roi_height))
            cropped_slice = slice_img[y0:y1, x0:x1]
            cropped_slices.append(cropped_slice)
        except Exception as e:
            print(f"Error processing slice {f}: {e}")
    if not cropped_slices:
        print(f"⚠️ No slices were successfully loaded and cropped. Skip this folder: {directory}")
        return None

    volume_array = np.stack(cropped_slices, axis=0)  # shape: (num_slices, height, width)
    # Crop in the z-dimension
    cropped_volume = volume_array[slice_range_start:slice_range_end, :, :]
    if cropped_volume.shape[0] == 0:
        print(f"⚠️ Empty volume after z-crop. Skip this folder: {directory}")
        return None

    return cropped_volume

def convert_numpy_to_sitk(volume_np):
    """Convert a NumPy array (z, y, x) to a SimpleITK image."""
    return sitk.GetImageFromArray(volume_np)

def rigid_registration(fixed_img, moving_img):
    """
    Perform rigid (Euler3D) registration between fixed and moving SimpleITK images.
    Returns the registered moving image.
    """
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.MetricUseFixedImageGradientFilterOff()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200, # 초기값은 learningrate=2.0 에 numberof iterations = 200
        gradientMagnitudeTolerance=1e-8)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_img, moving_img)

    print("Registration stopped:", registration_method.GetOptimizerStopConditionDescription())
    print("Final metric value:", registration_method.GetMetricValue())

    moving_registered = sitk.Resample(moving_img, fixed_img, final_transform,
                                      sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    return moving_registered

# Main code example
def main():
    # Fixed ROI parameters (for x,y cropping)
    ROI_x = 540
    ROI_y = 450
    ROI_width = 90
    ROI_height = 110
    slice_range_start = 260
    slice_range_end = 330 # Adjust as needed
    # Main folder containing subfolders with energy-specific data
    main_folder = '/media/joonho/2026 BNL/2026BNL/SC4 DOD13 0.1C Q 2week rest_1 RecO'
    output_folder = "/media/joonho/2026 BNL/2026BNL/SC4 DOD13 0.1C Q 2week rest_1 RecO/registered2"

    # recon_fly_scan_id_xxxxx 형태의 recon 폴더만 registration 대상으로 인식
    recon_prefix = "recon_fly_scan_id"
    subfolders = [
        os.path.join(main_folder, d)
        for d in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, d))
        and d.lower().startswith(recon_prefix)
    ]

    subfolders = sorted(subfolders)

    if not subfolders:
        print("No recon_fly_scan_id folders found.")
        return

    # For registration, choose the reference volume in the ORIGINAL recon-folder list.
    # 예: recon 폴더가 41개일 때 original_ref_index=30이면,
    # 0,2,4,...,40만 사용 후 selected ref_index는 15가 됨.
    original_ref_index = 12
    original_num_recon_folders = len(subfolders)

    # ============================================================
    # 핵심 수정:
    # recon_fly_scan_id 폴더가 41개이면 0.5 eV step 데이터로 보고,
    # 첫 번째, 세 번째, 다섯 번째 ... 즉 index 0,2,4,...,40만 사용.
    #
    # recon_fly_scan_id 폴더가 21개이면 기존처럼 전체 사용.
    # 그 외 개수이면 안전하게 기존처럼 전체 사용.
    # ============================================================
    if original_num_recon_folders == 41:
        selected_original_indices = list(range(0, original_num_recon_folders, 2))
        subfolders = [subfolders[i] for i in selected_original_indices]

        if original_ref_index in selected_original_indices:
            ref_index = selected_original_indices.index(original_ref_index)
        else:
            nearest_original_index = min(
                selected_original_indices,
                key=lambda x: abs(x - original_ref_index)
            )
            ref_index = selected_original_indices.index(nearest_original_index)

        print("✅ Detected 41 recon_fly_scan_id folders.")
        print("✅ Use odd-order recon folders only: 1st, 3rd, 5th, ... folders")
        print(f"✅ Original recon folder count = {original_num_recon_folders}")
        print(f"✅ Selected recon folder count = {len(subfolders)}")
        print(f"✅ Original ref_index = {original_ref_index}")
        print(f"✅ Converted selected ref_index = {ref_index}")

    else:
        ref_index = original_ref_index

        print(f"✅ recon_fly_scan_id folder count = {original_num_recon_folders}")
        print("✅ Use all recon folders without skipping.")

        if ref_index >= len(subfolders):
            print("Reference index out of range. Searching from the first folder as reference.")
            ref_index = 0

    # TIFF가 없는 폴더는 건너뛰고, 실제로 로드 가능한 첫 reference volume을 찾는다.
    ref_volume_np = None
    original_ref_index = ref_index

    for candidate_idx in list(range(ref_index, len(subfolders))) + list(range(0, ref_index)):
        print(f"Trying reference volume from: {subfolders[candidate_idx]}")
        ref_volume_np = load_and_crop_volume(
            subfolders[candidate_idx], ROI_x, ROI_y, ROI_width, ROI_height,
            slice_range_start, slice_range_end
        )
        if ref_volume_np is not None:
            ref_index = candidate_idx
            print(f"✅ Reference volume selected: index={ref_index}, folder={subfolders[ref_index]}")
            break

    if ref_volume_np is None:
        print("No valid reference volume found. All folders skipped.")
        return

    if original_ref_index != ref_index:
        print(f"Reference index changed from {original_ref_index} to {ref_index} because earlier folders had no TIFF.")

    ref_image = convert_numpy_to_sitk(ref_volume_np)

    # Store registered volumes (as numpy arrays) for each energy
    registered_volumes = []

    # Process each subfolder (each energy)
    for i, folder in enumerate(subfolders):
        print(f"\nProcessing folder {i + 1}: {folder}")
        try:
            vol_np = load_and_crop_volume(folder, ROI_x, ROI_y, ROI_width, ROI_height,
                                          slice_range_start, slice_range_end)
        except Exception as e:
            print(f"Error loading or cropping volume from {folder}: {e}")
            continue

        if vol_np is None:
            print(f"⛔ Skip folder without valid TIFF volume: {folder}")
            continue

        moving_image = convert_numpy_to_sitk(vol_np)
        if i == ref_index:
            reg_image = moving_image
            print("Using reference volume (no registration needed).")
        else:
            reg_image = rigid_registration(ref_image, moving_image)

        reg_np = sitk.GetArrayFromImage(reg_image)  # shape: (new_num_slices, height, width)
        registered_volumes.append(reg_np)

    if not registered_volumes:
        print("No volumes were successfully registered.")
        return

    num_energies = len(registered_volumes)
    num_slices_registered = registered_volumes[0].shape[0]
    print(f"Number of energies: {num_energies}, Number of slices per volume: {num_slices_registered}")

    # Instead of iterating over the absolute slice_range_start to slice_range_end,
    # iterate over the available slices in the registered (cropped) volume.

    os.makedirs(output_folder, exist_ok=True)
    for slice_idx in range(num_slices_registered):
        energy_stack_slice = np.stack([vol[slice_idx, :, :] for vol in registered_volumes], axis=0)
        output_filename = f"energy_stack_slice_{slice_idx:04d}.tif"
        output_path = os.path.join(output_folder, output_filename)
        imsave(output_path, energy_stack_slice)
        print(f"Saved energy stack for slice {slice_idx} to: {output_path}")

if __name__ == "__main__":
    main()
# src/slice_interpolation/sdt_interpolation.py

import numpy as np
from scipy.ndimage import distance_transform_edt


def mask_to_sdt(mask: np.ndarray) -> np.ndarray:
    """
    0/1 마스크 -> signed distance map (음수: 공동 내부, 양수: 외부)
    mask shape: (z, y)
    """
    mask_bool = mask.astype(bool)
    dist_out = distance_transform_edt(~mask_bool)  # 바깥 거리
    dist_in = distance_transform_edt(mask_bool)    # 안쪽 거리
    sdt = dist_out - dist_in
    return sdt.astype(np.float32)


def sdt_to_mask(sdt: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """
    signed distance map -> 0/1 마스크
    """
    return (sdt <= thresh).astype(np.uint8)


def interpolate_pair_sdt(
    sdt_A: np.ndarray,
    sdt_B: np.ndarray,
    num_mid: int
) -> np.ndarray:
    """
    두 SDT(sdt_A, sdt_B) 사이를 num_mid개의 중간 슬라이스로 채운다.
    반환: (num_mid + 2, z, y)  [순서: A, mid..., B]
    """
    assert sdt_A.shape == sdt_B.shape
    z, y = sdt_A.shape
    out = np.zeros((num_mid + 2, z, y), dtype=np.float32)
    out[0] = sdt_A
    out[-1] = sdt_B

    for i in range(1, num_mid + 1):
        t = i / (num_mid + 1)  # 0~1 사이
        out[i] = (1 - t) * sdt_A + t * sdt_B

    return out


def build_volume_from_masks(
    masks_zy,
    orig_spacing_x: float = 0.5,   # 원래 슬라이스 간격 (50cm)
    target_spacing_x: float = 0.1  # 보간 후 간격 (10cm)
):
    """
    여러 장의 (z,y) 마스크 리스트를 받아
    x 방향으로 SDT 보간을 수행하여 3D 볼륨(vol_zyx)을 만든다.

    masks_zy: list of np.ndarray, 각 shape = (z, y), 값은 0/1
    반환: (vol_zyx, num_mid)
        - vol_zyx: shape (z, y, x_new)
        - num_mid: 원본 슬라이스 한 쌍 사이에 생성된 중간 슬라이스 개수
    """
    n_seg = int(round(orig_spacing_x / target_spacing_x))  # 예: 0.5 / 0.1 = 5
    if n_seg < 1:
        raise ValueError("target_spacing_x가 orig_spacing_x보다 크면 안 됩니다.")
    num_mid = n_seg - 1  # 예: 5 - 1 = 4 (중간 슬라이스 개수)

    # 각 마스크당 SDT 미리 계산
    sdt_list = [mask_to_sdt(m) for m in masks_zy]

    all_sdt_slices = []
    for i in range(len(sdt_list) - 1):
        sdt_A = sdt_list[i]
        sdt_B = sdt_list[i + 1]
        pair_sdt = interpolate_pair_sdt(sdt_A, sdt_B, num_mid=num_mid)
        # 첫 쌍에서는 A~B 전체 사용, 이후부터는 앞 한 장(A) 빼고 붙임
        if i == 0:
            all_sdt_slices.append(pair_sdt)      # (num_mid+2, z, y)
        else:
            all_sdt_slices.append(pair_sdt[1:])  # 중복 A 제거

    all_sdt_slices = np.concatenate(all_sdt_slices, axis=0)  # (x_new, z, y)

    # 다시 0/1 마스크로
    masks_3d = sdt_to_mask(all_sdt_slices)  # (x_new, z, y)

    # 너가 쓰던 형식으로 transpose: (z, y, x)
    vol_zyx = np.transpose(masks_3d, (1, 2, 0))
    return vol_zyx, num_mid

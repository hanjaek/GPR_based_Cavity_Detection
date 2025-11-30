import numpy as np
from scipy.ndimage import distance_transform_edt


def mask_to_sdt(mask: np.ndarray) -> np.ndarray:
    """0/1 마스크 -> signed distance map (음수: 공동 내부, 양수: 외부)"""
    mask_bool = mask.astype(bool)
    dist_out = distance_transform_edt(~mask_bool)  # 바깥 거리
    dist_in = distance_transform_edt(mask_bool)    # 안쪽 거리
    sdt = dist_out - dist_in
    return sdt.astype(np.float32)


def sdt_to_mask(sdt: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """signed distance map -> 0/1 마스크"""
    return (sdt <= thresh).astype(np.uint8)


def interpolate_pair_sdt(sdt_A: np.ndarray,
                         sdt_B: np.ndarray,
                         num_mid: int) -> np.ndarray:
    """
    두 SDT 사이를 num_mid개의 중간 슬라이스로 채움.
    반환: (num_mid + 2, z, y)  [A, mid..., B]
    """
    assert sdt_A.shape == sdt_B.shape
    z, y = sdt_A.shape
    out = np.zeros((num_mid + 2, z, y), dtype=np.float32)
    out[0] = sdt_A
    out[-1] = sdt_B

    for i in range(1, num_mid + 1):
        t = i / (num_mid + 1)
        out[i] = (1 - t) * sdt_A + t * sdt_B

    return out


def build_volume_from_masks(masks_zy,
                            orig_spacing_x: float = 0.5,
                            target_spacing_x: float = 0.1):
    """
    여러 장 (z,y) 마스크 리스트를 받아
    x 방향 SDT 보간 후 vol_zyx (z,y,x) 생성.
    """
    n_seg = int(round(orig_spacing_x / target_spacing_x))  # 0.5 / 0.1 = 5
    if n_seg < 1:
        raise ValueError("target_spacing_x가 orig_spacing_x보다 크면 안 됩니다.")
    num_mid = n_seg - 1  # 중간 슬라이스 개수 = 5-1=4

    sdt_list = [mask_to_sdt(m) for m in masks_zy]

    all_sdt_slices = []
    for i in range(len(sdt_list) - 1):
        sdt_A = sdt_list[i]
        sdt_B = sdt_list[i + 1]
        pair_sdt = interpolate_pair_sdt(sdt_A, sdt_B, num_mid=num_mid)
        if i == 0:
            all_sdt_slices.append(pair_sdt)      # A~B 전부
        else:
            all_sdt_slices.append(pair_sdt[1:])  # 중복되는 A 제거

    all_sdt_slices = np.concatenate(all_sdt_slices, axis=0)  # (x_new, z, y)
    masks_3d = sdt_to_mask(all_sdt_slices)                   # (x_new, z, y)
    vol_zyx = np.transpose(masks_3d, (1, 2, 0))              # (z, y, x)
    return vol_zyx, num_mid

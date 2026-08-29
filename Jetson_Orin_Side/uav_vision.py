#!/usr/bin/env python3
import cv2
import numpy as np


EMPTY_RESULT = {
    "valid": False,
    "lateral_error": 0.0,
    "heading_error": 0.0,
    "confidence": 0.0,
    "road_width": 0.0,
    "road_center_x": 0.0,
    "image_center_x": 0.0,
    "coverage": 0.0,
    "fit_quality": 0.0,
    "solidity": 0.0,
}


def _weighted_robust_poly_fit(ys, xs, weights, height, degree=2, iterations=4, trim_frac=0.25):
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    min_points = degree + 2
    if ys.size < min_points:
        return None

    y_norm = ys / height
    keep = np.ones(ys.size, dtype=bool)
    coeffs = None

    for _ in range(iterations):
        if keep.sum() < min_points:
            break
        yk, xk, wk = y_norm[keep], xs[keep], weights[keep]
        try:
            coeffs = np.polyfit(yk, xk, degree, w=np.sqrt(wk))
        except (np.linalg.LinAlgError, ValueError):
            return None

        residuals_all = np.abs(xs - np.polyval(coeffs, y_norm))
        thresh = np.quantile(residuals_all[keep], 1.0 - trim_frac)
        keep = residuals_all <= max(thresh, 1e-6)

    if coeffs is None or keep.sum() < min_points:
        return None

    final_residuals = xs[keep] - np.polyval(coeffs, y_norm[keep])
    rmse = float(np.sqrt(np.mean(final_residuals ** 2)))
    return coeffs, rmse, int(keep.sum())


def extract_road_information(mask, debug=False, debug_path="road_extraction_debug.png",
                              n_sample_rows=48, roi_top_frac=0.05, roi_bottom_frac=0.97,
                              min_row_pixels=5, min_road_area=500,
                              bridge_kernel_size=25, clean_kernel_size=7,
                              poly_degree=2):

    if mask is None:
        return dict(EMPTY_RESULT)

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    height, width = mask.shape
    image_center_x = width / 2.0

    #  Fine cleanup
    clean_kernel = np.ones((clean_kernel_size, clean_kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, clean_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, clean_kernel)

    # Gap-bridging pass to recover the road split by occlusion 
    bridge_kernel = np.ones((bridge_kernel_size, bridge_kernel_size), np.uint8)
    bridged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, bridge_kernel)

    contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dict(EMPTY_RESULT)

    largest_contour = max(contours, key=cv2.contourArea)
    road_area = cv2.contourArea(largest_contour)
    if road_area < min_road_area:
        return dict(EMPTY_RESULT)

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(road_area / hull_area) if hull_area > 0 else 0.0

    selection_mask = np.zeros_like(mask)
    cv2.drawContours(selection_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    road_mask = cv2.bitwise_and(cleaned, selection_mask)

    #3. Dense row sampling 
    _, contour_y, _, contour_h = cv2.boundingRect(largest_contour)
    contour_y_min = contour_y
    contour_y_max = contour_y + contour_h

    cap_top = int(height * roi_top_frac)
    cap_bottom = int(height * roi_bottom_frac)

    band_top = max(contour_y_min, cap_top)
    band_bottom = min(contour_y_max, cap_bottom)

    if band_bottom <= band_top:
        result = dict(EMPTY_RESULT)
        result["solidity"] = solidity
        return result

    sample_rows = np.linspace(band_top, band_bottom, n_sample_rows).astype(int)

    ys, xs, weights, row_widths = [], [], [], []
    for y in sample_rows:
        row_pixels = np.nonzero(road_mask[y])[0]
        if row_pixels.size < min_row_pixels:
            continue
        ys.append(float(y))
        xs.append(float(np.median(row_pixels)))
        weights.append(float(row_pixels.size))
        lo, hi = np.percentile(row_pixels, [5, 95])
        row_widths.append(float(hi - lo))

    n_attempted = len(sample_rows)
    n_found = len(ys)
    coverage = n_found / n_attempted if n_attempted else 0.0

    min_points_needed = poly_degree + 2
    if n_found < max(5, min_points_needed):
        result = dict(EMPTY_RESULT)
        result["coverage"] = coverage
        result["solidity"] = solidity
        return result

    fit = _weighted_robust_poly_fit(ys, xs, weights, height, degree=poly_degree)
    if fit is None:
        result = dict(EMPTY_RESULT)
        result["coverage"] = coverage
        result["solidity"] = solidity
        return result

    coeffs, rmse, n_kept = fit

    road_width = float(np.median(row_widths)) if row_widths else 0.0
    half_width = max(road_width / 2.0, 1.0)
    fit_quality = float(1.0 / (1.0 + (rmse / half_width) ** 2))
    solidity_term = 0.5 + 0.5 * min(solidity, 1.0)
    confidence = float(np.clip(coverage * fit_quality * solidity_term, 0.0, 1.0))

    # Lateral / heading error 
    y_near = ys[-1]                     
    y_near_norm = y_near / height
    current_center_x = float(np.polyval(coeffs, y_near_norm))

    lateral_error = (current_center_x - image_center_x) / (width / 2.0)
    lateral_error = float(np.clip(lateral_error, -1.0, 1.0))


    deriv_coeffs = np.polyder(coeffs)
    slope_norm = float(np.polyval(deriv_coeffs, y_near_norm))
    slope_px = slope_norm / height   # d(x_px)/d(y_norm) -> d(x_px)/d(y_px)
    heading_error = float(np.degrees(np.arctan(slope_px)))

    if debug:
        _draw_debug(road_mask, ys, xs, coeffs, height, image_center_x, debug_path)

    return {
        "valid": True,
        "lateral_error": lateral_error,
        "heading_error": heading_error,
        "confidence": confidence,
        "road_width": road_width,
        "road_center_x": current_center_x,
        "image_center_x": float(image_center_x),
        "coverage": float(coverage),
        "fit_quality": float(fit_quality),
        "solidity": float(solidity),
    }


def _draw_debug(road_mask, ys, xs, coeffs, height, image_center_x, path):
    debug = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    cv2.line(debug, (int(image_center_x), 0), (int(image_center_x), height), (255, 0, 0), 2)

    for x, y in zip(xs, ys):
        cv2.circle(debug, (int(x), int(y)), 6, (0, 255, 0), -1)

    curve_ys = np.linspace(0, height - 1, 200)
    curve_xs = np.polyval(coeffs, curve_ys / height)
    curve_pts = np.stack([curve_xs, curve_ys], axis=1).astype(np.int32)
    cv2.polylines(debug, [curve_pts], isClosed=False, color=(0, 0, 255), thickness=3)

    cv2.imwrite(path, debug)

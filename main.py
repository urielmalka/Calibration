import glob
import cv2
import numpy as np
import json

def calibrate_and_save_json(
    images_glob,
    output_json,
    pattern_size=(28, 17),
    square_size=10.0
):
    cols, rows = pattern_size

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(images_glob))
    if not images:
        raise FileNotFoundError("No images found")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    image_size = None
    used = 0

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w,h)

        ret, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used < 8:
        raise RuntimeError("Not enough valid images")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # RMSE
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        total_err += err * err
        total_pts += len(proj)

    rmse = (total_err / total_pts) ** 0.5

    # JSON-friendly dict
    data = {
        "image_size": list(image_size),
        "square_size_mm": square_size,
        "pattern_size": list(pattern_size),
        "used_images": used,
        "rmse_pixels": float(rmse),
        "K": K.tolist(),
        "distortion": dist.flatten().tolist()
    }

    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Calibration saved to {output_json}")
    return data


if __name__ == "__main__":
    calibrate_and_save_json(
        images_glob="images/mono/iphone17/*.jpg",   
        output_json="intrinsic.json",
        pattern_size=(28, 17),
        square_size=10.0
    )

import math
import numpy as np
from math import radians
from PIL import Image
from numba import njit, prange
import qrcode
from PIL import Image
import csv
import itertools
import os
from skimage.transform import resize

def generate_qr_code_array(data, version=1, error_correction='L', box_size=10, border=0, fill_color="black", back_color="white"):
    """Generates a QR code for the given data and returns it as a NumPy array.

    Args:
        data (str): The data to encode in the QR code.
        version (int, optional): The version of the QR code (1-40). Defaults to 1.
        error_correction (str, optional): The error correction level ('L', 'M', 'Q', 'H'). Defaults to 'L'.
        box_size (int, optional): The size of each box (pixel) in the QR code. Defaults to 10.
        border (int, optional): The width of the border around the QR code. Defaults to 0.
        fill_color (str, optional): The color of the QR code modules. Defaults to "black".
        back_color (str, optional): The background color of the QR code. Defaults to "white".

    Returns:
        numpy.ndarray: A NumPy array representing the QR code image (0 for background, 255 for foreground).
    """
    try:
        qr = qrcode.QRCode(
            version=version,
            error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{error_correction.upper()}'),
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color=fill_color, back_color=back_color).convert('L')  # Convert to grayscale

        # Convert the PIL Image to a NumPy array
        img_array = np.array(img)

        return img_array
    except AttributeError:
        raise ValueError(f"Invalid error_correction level: {error_correction}. Choose from 'L', 'M', 'Q', 'H'.")
    except ImportError:
        print("Error: The 'qrcode' and 'Pillow' libraries are required. Please install them using:\n\npip install qrcode Pillow")
        return None

@njit
def normalize_nb(v0, v1, v2):
    """Return the normalized vector of v."""
    length = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    if length < 1e-12:
        return 0.0, 0.0, 0.0
    return v0 / length, v1 / length, v2 / length

@njit
def rotate_y_nb(x, y, z, angle_degrees):
    """
    Rotate vector (x, y, z) around the Y-axis by angle_degrees.
    """
    angle = radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_new = x * cos_a + z * sin_a
    y_new = y
    z_new = -x * sin_a + z * cos_a
    return x_new, y_new, z_new

@njit(parallel=True)
def render_sphere_kernel(
    output_array,
    qr_array,
    sphere_diameter_mm,
    qr_side_length_mm,
    camera_distance_mm,
    focal_length_mm,
    sphere_rotation_degrees,
    camera_width_pixels,
    camera_height_pixels,
    sensor_width_mm,
    sensor_height_mm
):
    """
    Numba-accelerated function to render the sphere with QR code. All length units are in millimeters.
    """
    qr_size = qr_array.shape[0]
    radius_mm = sphere_diameter_mm / 2.0

    aspect_ratio = camera_width_pixels / camera_height_pixels
    # Adjust sensor width based on aspect ratio to maintain the correct field of view
    # if the rendered image aspect ratio differs from the sensor aspect ratio.
    # However, here we map directly to pixel coordinates, so sensor dimensions are crucial.

    # Sphere center in world coordinates (mm)
    sphere_center_x = 0.0
    sphere_center_y = 0.0
    sphere_center_z = camera_distance_mm - focal_length_mm

    # Camera origin in world coordinates (mm)
    cam_origin_x = 0.0
    cam_origin_y = 0.0
    cam_origin_z = -focal_length_mm

    # Half the physical size of the QR sticker on the sphere (linear dimension, mm)
    half_qr_length_mm = qr_side_length_mm / 2.0

    # Angle subtended by half the QR sticker at the center of the sphere
    half_angle = half_qr_length_mm / radius_mm  # Use arc length directly

    for py in prange(camera_height_pixels):
        # Map pixel coordinates to normalized sensor coordinates [-1, 1]
        ndc_y = (py + 0.5) / camera_height_pixels  # [0,1]
        sensor_y = (1.0 - 2.0 * ndc_y) * (sensor_height_mm / 2.0)

        for px in range(camera_width_pixels):
            # Map pixel coordinates to normalized sensor coordinates [-1, 1]
            ndc_x = (px + 0.5) / camera_width_pixels  # [0,1]
            sensor_x = (2.0 * ndc_x - 1.0) * (sensor_width_mm / 2.0)

            sensor_z = 0.0 # Sensor plane is at z=0 in camera space

            # Ray direction from camera origin to sensor point (mm)
            rx = sensor_x - cam_origin_x
            ry = sensor_y - cam_origin_y
            rz = sensor_z - cam_origin_z

            # Normalize ray direction
            rd_x, rd_y, rd_z = normalize_nb(rx, ry, rz)

            # Ray-sphere intersection (all distances in mm)
            ox = cam_origin_x - sphere_center_x
            oy = cam_origin_y - sphere_center_y
            oz = cam_origin_z - sphere_center_z

            a = rd_x * rd_x + rd_y * rd_y + rd_z * rd_z
            b = 2.0 * (ox * rd_x + oy * rd_y + oz * rd_z)
            c = ox * ox + oy * oy + oz * oz - radius_mm * radius_mm

            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                # No intersection; set background to white
                output_array[py, px, 0] = 255
                output_array[py, px, 1] = 255
                output_array[py, px, 2] = 255
                continue

            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            t_hit = -1.0
            if t1 > 0.0 and t2 > 0.0:
                t_hit = t1 if t1 < t2 else t2
            elif t1 > 0.0:
                t_hit = t1
            elif t2 > 0.0:
                t_hit = t2

            if t_hit < 0.0:
                # Intersection behind camera; set background to white
                output_array[py, px, 0] = 255
                output_array[py, px, 1] = 255
                output_array[py, px, 2] = 255
                continue

            # Compute intersection point (mm)
            ix = cam_origin_x + t_hit * rd_x
            iy = cam_origin_y + t_hit * rd_y
            iz = cam_origin_z + t_hit * rd_z

            # Compute normal at intersection
            nx = ix - sphere_center_x
            ny = iy - sphere_center_y
            nz = iz - sphere_center_z
            nx, ny, nz = normalize_nb(nx, ny, nz)

            # Rotate normal to simulate sphere rotation
            nx, ny, nz = rotate_y_nb(nx, ny, nz, sphere_rotation_degrees)

            # Convert normal to spherical coordinates relative to +Z axis
            lon = math.atan2(nx, nz)  # Longitude
            lat = math.asin(ny)       # Latitude

            # Check if the normal is within the QR sticker region
            if abs(lon) < half_angle and abs(lat) < half_angle:
                # Map (lon, lat) to [0,1] x [0,1]
                fraction_lon = (lon + half_angle) / (2.0 * half_angle)
                fraction_lat = (lat + half_angle) / (2.0 * half_angle)

                # Convert to QR image pixel coordinates
                qr_u = int(fraction_lon * qr_size)
                qr_v = int((1.0 - fraction_lat) * qr_size)  # Invert Y

                # Boundary check
                if qr_u >= 0 and qr_u < qr_size and qr_v >= 0 and qr_v < qr_size:
                    gray_value = qr_array[qr_v, qr_u]
                    output_array[py, px, 0] = gray_value
                    output_array[py, px, 1] = gray_value
                    output_array[py, px, 2] = gray_value
                else:
                    # Outside QR range; shade as gray
                    shade = 180
                    output_array[py, px, 0] = shade
                    output_array[py, px, 1] = shade
                    output_array[py, px, 2] = shade
            else:
                # Outside sticker region; apply simple Lambert shading
                light_dir_x = 0.0
                light_dir_y = 0.0
                light_dir_z = 1.0
                ndotl = nx * light_dir_x + ny * light_dir_y + nz * light_dir_z
                if ndotl < 0.0:
                    ndotl = 0.0
                shade = int(50 + 200 * ndotl)
                if shade > 255:
                    shade = 255
                elif shade < 0:
                    shade = 0
                output_array[py, px, 0] = shade
                output_array[py, px, 1] = shade
                output_array[py, px, 2] = shade

def render_sphere_with_qr(
    sphere_diameter_mm,
    qr_side_length_mm,
    camera_distance_mm,
    focal_length_mm,
    sphere_rotation_degrees,
    camera_width_pixels,
    camera_height_pixels,
    output_dir,
    filename,
    sensor_width_mm,
    sensor_height_mm
):
    """
    Render a sphere with a QR code sticker using Numba for acceleration.
    All length units are in millimeters.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the QR code as a NumPy array
    data_to_encode = "www.scionresearch.com"
    qr_array = generate_qr_code_array(data_to_encode)

    # Initialize the output image array (height, width, 3)
    output_array = np.zeros((camera_height_pixels, camera_width_pixels, 3), dtype=np.uint8)

    # Call the Numba-accelerated rendering kernel
    render_sphere_kernel(
        output_array,
        qr_array,
        sphere_diameter_mm,
        qr_side_length_mm,
        camera_distance_mm,
        focal_length_mm,
        sphere_rotation_degrees,
        camera_width_pixels,
        camera_height_pixels,
        sensor_width_mm,
        sensor_height_mm
    )

    # Generate filename if not provided
    if filename is None:
        filename = f"sphere_diameter_{sphere_diameter_mm:.2f}mm.png"
    
    # Combine directory and filename
    full_path = os.path.join(output_dir, filename)

    print(f"Size of full image: {output_array.shape}")

    # Downsample and crop the image to match the viewfinder of the phone
    if simulate_device_viewfinder == True:
        # Downsample the image to an equivalent resolution
        downsample_width_pixels = device_display_width_pixels
        downsample_ratio = downsample_width_pixels / camera_width_pixels
        downsample_height_pixels = int(downsample_ratio * camera_height_pixels)
        downsampled_image = resize(output_array, (downsample_height_pixels, downsample_width_pixels, 3), anti_aliasing=True)

        # Convert from float [0,1] to uint8 [0,255]
        downsampled_image = (downsampled_image * 255).astype(np.uint8)  # Correct conversion

        import matplotlib.pyplot as plt
        plt.imshow(downsampled_image)
        plt.show()

        print(f"Size of downsampled image: {downsampled_image.shape}")
        print(f"Type of downsampled image: {type(downsampled_image)}")

        # Crop the image to the correct aspect ratio
        output_img = Image.fromarray(downsampled_image, 'RGB')

    elif simulate_device_viewfinder == False:
        # Convert the NumPy array to a PIL Image
        output_img = Image.fromarray(output_array, 'RGB')
    
    # Save the image
    output_img.save(full_path, "PNG")
    print(f"Saved {full_path}")
    return full_path  # Return full path

###################### SETTINGS ########################

# Parameters to iterate through
diameters_mm = [50]
qr_side_lengths_mm = [21]
camera_distances_mm = [200]

# Set your custom output folder here
output_directory = r"Images"  # ← Change this to your desired folder

# Camera parameters
camera_orientation = "portrait" # Orientation of the camera, options are "portrait" or "landscape"
focal_length_mm = 5.7 # The true focal length (mm) of the camera, not the 35mm equivalent. This value can be found by uploading a photo from the camera to an EXIF data viewer such as www.jimpl.com and looking for the "Focal length" setting.
sphere_rotation_degrees = 180.0 # Rotation of the sphere around a vertical axis passing through it. Default at 180 degrees has the QR code directly facing the camera
camera_width_pixels = 3024 # Number of pixels in the width of a photo that the camera can produce
camera_height_pixels = 4032 # Number of pixels in the height of a photo the the camera can produce
sensor_width_mm = 5.7456 # Width of the physical sensor. This can be found in various ways. On Android, the app "Device Info HW" from the Google Play store has this information. Alternatively, if you know the pixel size (in micrometres), multiply that by the number of pixels in length and width of the photo.
sensor_height_mm = 7.6608
simulate_device_viewfinder = True # Use if you want to simulate the image that the viewfinder of a phone would actually see (i.e. only the pixels that are displayed on the screen, not all available camera pixels). This is what the scanning apps on the phones can see.
device_display_width_pixels = 2778 # The number of pixels that make up the phone display
device_display_height_pixels = 1284
viewfinder_aspect_ratio = 4/3 # Aspect ratio that the viewfinder is set to on the phone e.g. 4:3 --> 4/3

csv_file = "rendered_spheres_mm_TEMP.csv"

################### END OF SETTINGS #####################

# Check that orientation of sensor w&h and image w&h matches camera orientation
if camera_orientation == 'portrait':
    if camera_width_pixels > camera_height_pixels:
        camera_width_pixels, camera_height_pixels = camera_height_pixels, camera_width_pixels
elif camera_orientation == 'landscape':
    if camera_width_pixels < camera_height_pixels:
        camera_width_pixels, camera_height_pixels = camera_height_pixels, camera_width_pixels

if camera_orientation == 'portrait':
    if sensor_width_mm > sensor_height_mm:
        sensor_width_mm, sensor_height_mm = sensor_height_mm, sensor_width_mm
elif camera_orientation == 'landscape':
    if sensor_width_mm < sensor_height_mm:
        sensor_width_mm, sensor_height_mm = sensor_height_mm, sensor_width_mm

if camera_orientation == 'portrait':
    if device_display_width_pixels > device_display_height_pixels:
        device_display_width_pixels, device_display_height_pixels = device_display_height_pixels, device_display_width_pixels
elif camera_orientation == 'landscape':
    if device_display_width_pixels < device_display_height_pixels:
       device_display_width_pixels, device_display_height_pixels = device_display_height_pixels, device_display_width_pixels


fieldnames = ['diameter_mm', 'qr_side_length_mm', 'camera_distance_mm', 
                'focal_length_mm', 'sphere_rotation_degrees', 
                'camera_width_pixels', 'camera_height_pixels', 'filename']

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through all parameter combinations
    for diameter_mm, qr_side_mm, camera_dist_mm in itertools.product(
        diameters_mm, qr_side_lengths_mm, camera_distances_mm
    ):
        # Generate filename with all parameters
        filename = (
            f"sphere_d_{diameter_mm:.1f}mm_"
            f"qr_{qr_side_mm:.1f}mm_"
            f"cam_{camera_dist_mm:.1f}mm.png"
        )
        
        # Render the sphere with current parameters
        full_path = render_sphere_with_qr(
            sphere_diameter_mm=diameter_mm,
            qr_side_length_mm=qr_side_mm,
            camera_distance_mm=camera_dist_mm,
            focal_length_mm=focal_length_mm,
            sphere_rotation_degrees=sphere_rotation_degrees,
            camera_width_pixels=camera_width_pixels,
            camera_height_pixels=camera_height_pixels,
            output_dir=output_directory,
            filename=filename,
            sensor_width_mm = sensor_width_mm,
            sensor_height_mm = sensor_height_mm
        )
        
        # Write parameters to CSV
        writer.writerow({
            'diameter_mm': diameter_mm,
            'qr_side_length_mm': qr_side_mm,
            'camera_distance_mm': camera_dist_mm,
            'focal_length_mm': focal_length_mm,
            'sphere_rotation_degrees': sphere_rotation_degrees,
            'camera_width_pixels': camera_width_pixels,
            'camera_height_pixels': camera_height_pixels,
            'filename': full_path  # Record full path in CSV
        })

print(f"Parameters and filenames recorded in {csv_file}")
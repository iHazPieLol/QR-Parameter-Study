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

def generate_qr_code_array(data, version=1, error_correction='L', box_size=10, border=1, fill_color="black", back_color="white"):
    """Generates a QR code for the given data and returns it as a NumPy array.

    Args:
        data (str): The data to encode in the QR code.
        version (int, optional): The version of the QR code (1-40). Defaults to 1.
        error_correction (str, optional): The error correction level ('L', 'M', 'Q', 'H'). Defaults to 'L'.
        box_size (int, optional): The size of each box (pixel) in the QR code. Defaults to 10.
        border (int, optional): The width of the border around the QR code. Defaults to 1.
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
    qr_size = qr_array.shape[0]
    radius_mm = sphere_diameter_mm / 2.0

    # Sphere center in world coordinates (mm)
    sphere_center_x = 0.0
    sphere_center_y = 0.0
    sphere_center_z = camera_distance_mm - focal_length_mm

    # Camera origin in world coordinates (mm)
    cam_origin_x = 0.0
    cam_origin_y = 0.0
    cam_origin_z = -focal_length_mm

    # QR sticker angular size calculations
    half_qr_length_mm = qr_side_length_mm / 2.0
    theta = half_qr_length_mm / radius_mm  # Angular half-size (radians)
    half_angle = math.tan(theta)  # Projection scale factor

    for py in prange(camera_height_pixels):
        ndc_y = (py + 0.5) / camera_height_pixels
        sensor_y = (1.0 - 2.0 * ndc_y) * (sensor_height_mm / 2.0)

        for px in range(camera_width_pixels):
            ndc_x = (px + 0.5) / camera_width_pixels
            sensor_x = (2.0 * ndc_x - 1.0) * (sensor_width_mm / 2.0)

            # Ray direction calculation
            rx = sensor_x - cam_origin_x
            ry = sensor_y - cam_origin_y
            rz = 0.0 - cam_origin_z
            rd_x, rd_y, rd_z = normalize_nb(rx, ry, rz)

            # Ray-sphere intersection
            ox = cam_origin_x - sphere_center_x
            oy = cam_origin_y - sphere_center_y
            oz = cam_origin_z - sphere_center_z

            a = rd_x**2 + rd_y**2 + rd_z**2
            b = 2.0 * (ox*rd_x + oy*rd_y + oz*rd_z)
            c = ox**2 + oy**2 + oz**2 - radius_mm**2
            disc = b**2 - 4.0*a*c

            if disc < 0.0:
                output_array[py, px] = 255
                continue

            sqrt_disc = math.sqrt(disc)
            t_hit = (-b - sqrt_disc) / (2.0*a)
            if t_hit < 0.0:
                t_hit = (-b + sqrt_disc) / (2.0*a)
                if t_hit < 0.0:
                    output_array[py, px] = 255
                    continue

            # Intersection point and normal
            ix = cam_origin_x + t_hit*rd_x
            iy = cam_origin_y + t_hit*rd_y
            iz = cam_origin_z + t_hit*rd_z
            nx, ny, nz = normalize_nb(ix - sphere_center_x, iy - sphere_center_y, iz - sphere_center_z)
            nx, ny, nz = rotate_y_nb(nx, ny, nz, sphere_rotation_degrees)

            # Front-face check and tangent projection
            if nz <= 1e-6:  # Back face or edge case
                ndotl = max(0.0, nz)
                shade = int(50 + 200*ndotl)
                output_array[py, px] = min(max(shade, 0), 255)
                continue

            # Project onto tangent plane at (0,0,1)
            tx = nx / nz
            ty = ny / nz

            # QR region check
            if abs(tx) <= half_angle and abs(ty) <= half_angle:
                # Map to QR coordinates
                u = (tx + half_angle) / (2*half_angle)
                v = (half_angle - ty) / (2*half_angle)  # Flip Y-axis
                qr_u = int(u * (qr_size-1))
                qr_v = int(v * (qr_size-1))

                # Clamp and sample
                qr_u = max(0, min(qr_size-1, qr_u))
                qr_v = max(0, min(qr_size-1, qr_v))
                gray = qr_array[qr_v, qr_u]
                output_array[py, px] = gray
            else:
                # Lambert shading
                ndotl = max(0.0, nz)
                shade = int(50 + 200*ndotl)
                output_array[py, px] = min(max(shade, 0), 255)

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
camera_distances_mm = [100]

# Set your custom output folder here
output_directory = r"Images"  # ← Change this to your desired folder

# Camera parameters --> Can add information for new devices into here
device_parameters = {
    'iPhone_13_Pro_Max_Main' : {
        'focal_length_mm' : 5.7,
        'camera_width_pixels' : 3024,
        'camera_height_pixels' : 4032,
        'sensor_width_mm' : 5.7456,
        'sensor_height_mm' : 7.6608,
        'device_display_width_pixels' : 2778,
        'device_display_height_pixels' : 1284
    },
    'iPhone_13_Pro_Max_Ultrawide': {
        'focal_length_mm': 1.57,
        'camera_width_pixels': 3024,
        'camera_height_pixels': 4032,
        'sensor_width_mm': 3.024,
        'sensor_height_mm': 4.032,
        'device_display_width_pixels': 2778,
        'device_display_height_pixels': 1284
    }
}
camera_orientation = "portrait" # Orientation of the camera, options are "portrait" or "landscape"
sphere_rotation_degrees = 180.0 # Rotation of the sphere around a vertical axis passing through it. Default at 180 degrees has the QR code directly facing the camera
simulate_device_viewfinder = True # Use if you want to simulate the image that the viewfinder of a phone would actually see (i.e. only the pixels that are displayed on the screen, not all available camera pixels). This is what the scanning apps on the phones can see.
device = 'iPhone_13_Pro_Max_Ultrawide' # Name of device (must be in device_parameters)
viewfinder_aspect_ratio = 4/3 # Aspect ratio that the viewfinder is set to on the phone e.g. 4:3 --> 4/3
digital_zoom = 1.909090909

csv_file = "rendered_spheres_mm_TEMP.csv"

################### END OF SETTINGS #####################

################# DESCRIPTIONS OF SETTINGS #######################

# focal_length_mm --> The true focal length (mm) of the camera, not the 35mm equivalent. This value can be found by uploading a photo from the camera to an EXIF data viewer such as www.jimpl.com and looking for the "Focal length" setting.
# camera_width_pixels --> Number of pixels in the width of a photo that the camera can produce
# camera_height_pixels --> Number of pixels in the height of a photo the the camera can produce
# sensor_width_mm --> Width of the physical sensor. This can be found in various ways. On Android, the app "Device Info HW" from the Google Play store has this information. Alternatively, if you know the pixel size (in micrometres), multiply that by the number of pixels in length and width of the photo.
# sensor_height_mm --> Same as above, but for the sensor height.
# device_display_width_pixels --> The number of pixels that make up the phone display
# device_display_height_pixels --> Similar to above

################## END OF DESCRIPTIONS OF SETTINGS ###################

# Get correct information from chosen device
focal_length_mm = device_parameters[device]['focal_length_mm']
camera_width_pixels = device_parameters[device]['camera_width_pixels']
camera_height_pixels = device_parameters[device]['camera_height_pixels']
sensor_width_mm = device_parameters[device]['sensor_width_mm']
sensor_height_mm = device_parameters[device]['sensor_height_mm']
device_display_width_pixels = device_parameters[device]['device_display_width_pixels']
device_display_height_pixels = device_parameters[device]['device_display_height_pixels']

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
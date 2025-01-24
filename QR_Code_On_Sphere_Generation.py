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
import matplotlib.pyplot as plt

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
    sensor_height_mm,
    noise_level,
    ambient_light_intensity,
    diffuse_light_intensity,
    specular_light_intensity,
    specular_exponent
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

            # Ray direction calculation (view vector)
            rx = sensor_x - cam_origin_x
            ry = sensor_y - cam_origin_y
            rz = 0.0 - cam_origin_z
            vx, vy, vz = normalize_nb(rx, ry, rz)  # View vector

            # Ray-sphere intersection
            ox = cam_origin_x - sphere_center_x
            oy = cam_origin_y - sphere_center_y
            oz = cam_origin_z - sphere_center_z

            a = vx**2 + vy**2 + vz**2  # Should be 1.0 for normalized vectors
            b = 2.0 * (ox*vx + oy*vy + oz*vz)
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
            ix = cam_origin_x + t_hit*vx
            iy = cam_origin_y + t_hit*vy
            iz = cam_origin_z + t_hit*vz
            nx, ny, nz = normalize_nb(ix - sphere_center_x, iy - sphere_center_y, iz - sphere_center_z)
            nx, ny, nz = rotate_y_nb(nx, ny, nz, sphere_rotation_degrees)

            # Lighting Calculation
            # Light direction (assume light source at camera position)
            lx, ly, lz = normalize_nb(-cam_origin_x, -cam_origin_y, -cam_origin_z)

            # Reflection vector
            rx, ry, rz = 2.0*(nx*lx + ny*ly + nz*lz)*nx - lx, 2.0*(nx*lx + ny*ly + nz*lz)*ny - ly, 2.0*(nx*lx + ny*ly + nz*lz)*nz - lz

            # Diffuse component
            ndotl = max(0.0, nx*lx + ny*ly + nz*lz)
            diffuse = diffuse_light_intensity * ndotl

            # Specular component (Phong model)
            rdotv = max(0.0, rx*vx + ry*vy + rz*vz)
            specular = specular_light_intensity * (rdotv ** specular_exponent)

            # Ambient component
            ambient = ambient_light_intensity

            # Total light intensity
            light_intensity = ambient + diffuse + specular

            # Front-face check and tangent projection
            if nz <= 1e-6:  # Back face or edge case
                shade = int(255 * light_intensity)

                # Add noise
                noise = np.random.normal(0, noise_level)
                shade = int(shade + noise)

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

                # Specular component does not affect QR code
                qr_light_intensity = ambient + diffuse
                gray = qr_array[qr_v, qr_u] * qr_light_intensity  # QR code brightness

                # Add noise
                noise = np.random.normal(0, noise_level)
                gray = int(gray + noise)

                output_array[py, px] = min(max(gray, 0), 255)

            else:
                # Shading with specular highlights
                shade = int(255 * light_intensity)

                # Add noise
                noise = np.random.normal(0, noise_level)
                shade = int(shade + noise)

                output_array[py, px] = min(max(shade, 0), 255)

@njit
def crop_center(img, crop_width, crop_height):
    """Crops an image around its center.

    Args:
        img (np.ndarray): The image array (height, width, channels).
        crop_width (int): The desired width of the cropped image.
        crop_height (int): The desired height of the cropped image.

    Returns:
        np.ndarray: The cropped image.
    """

    height, width = img.shape[0:2] # Get height and width from the first two dimensions
    center_x = width // 2
    center_y = height // 2

    start_x = center_x - crop_width // 2
    start_y = center_y - crop_height // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height

    #Handle edges to ensure we are not out of bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(width, end_x)
    end_y = min(height, end_y)

    cropped_img = img[start_y:end_y, start_x:end_x]

    return cropped_img

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
    sensor_height_mm,
    noise_level,
    ambient_light_intensity,
    diffuse_light_intensity,
    specular_light_intensity,
    specular_exponent
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
        sensor_height_mm,
        noise_level,
        ambient_light_intensity,
        diffuse_light_intensity,
        specular_light_intensity,
        specular_exponent
    )

    # Generate filename if not provided
    if filename is None:
        filename = f"sphere_diameter_{sphere_diameter_mm:.2f}mm.png"
    
    # Combine directory and filename
    full_path = os.path.join(output_dir, filename)

    print(f"Size of full image: {output_array.shape}")

    # Perform digital zooming by cropping the image and then upscaling it
    output_array = crop_center(output_array, crop_width=int(camera_width_pixels / digital_zoom), crop_height=int(camera_height_pixels / digital_zoom)) # Crop the image
    output_array = resize(output_array, (camera_width_pixels, camera_height_pixels)) # Upsample it back to original size
    output_array = (output_array * 255).astype(np.uint8) # Scale array back to 0-255 rgb range

    # Downsample and crop the image to match the viewfinder of the phone
    if simulate_device_viewfinder == True:
        # Downsample the image to an equivalent resolution
        downsample_width_pixels = device_display_width_pixels
        downsample_ratio = downsample_width_pixels / camera_width_pixels
        downsample_height_pixels = int(downsample_ratio * camera_height_pixels)
        downsampled_image = resize(output_array, (downsample_height_pixels, downsample_width_pixels, 3), anti_aliasing=True)

        # Convert from float [0,1] to uint8 [0,255]
        downsampled_image = (downsampled_image * 255).astype(np.uint8)  # Correct conversion

        if show_image == True:
            plt.imshow(downsampled_image)
            plt.show()

        # Crop the image to the correct aspect ratio
        #   Implement this...

        output_img = Image.fromarray(downsampled_image, 'RGB')

    elif simulate_device_viewfinder == False:
        
        if show_image == True:
            plt.imshow(output_array)
            plt.show()
        
        # Convert the NumPy array to a PIL Image
        output_img = Image.fromarray(output_array, 'RGB')
    
    # Save the image
    if export_image == True:
        output_img.save(full_path, "PNG")
        print(f"Saved {full_path}")
        return full_path  # Return full path

###################### SETTINGS ########################

# Parameters to iterate through
diameters_mm = [50]
qr_side_lengths_mm = [21]
camera_distances_mm = [100]
noise_levels = [0]

ambient_light_intensities = [0.4]
diffuse_light_intensities = [0.6]
specular_light_intensities = [0.5]  # Control the brightness of the highlight
specular_exponents = [15]        # Control the size of the highlight (smaller = larger highlight)

# Set your custom output folder here
output_directory = r"Images"

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
show_image = True # Open a window to display the generated image? Needs to be closed in order for the rest of the script to keep runnning
export_image = False # Export the generated image?
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
              'camera_width_pixels', 'camera_height_pixels',
              'noise_level', 'ambient_light_intensity',
              'diffuse_light_intensity', 'specular_light_intensity',
              'specular_exponent', 'filename']

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for diameter_mm, qr_side_mm, camera_dist_mm, noise_level, ambient_light_intensity, diffuse_light_intensity, specular_light_intensity, specular_exponent in itertools.product(
            diameters_mm, qr_side_lengths_mm, camera_distances_mm, noise_levels, ambient_light_intensities, diffuse_light_intensities, specular_light_intensities, specular_exponents
        ):
            filename = (
                f"sphere_d_{diameter_mm:.1f}mm_"
                f"qr_{qr_side_mm:.1f}mm_"
                f"cam_{camera_dist_mm:.1f}mm_"
                f"noise_{noise_level:.1f}_"
                f"ambient_{ambient_light_intensity:.1f}_"
                f"diffuse_{diffuse_light_intensity:.1f}_"
                f"specular_{specular_light_intensity:.1f}_"
                f"exponent_{specular_exponent:.1f}.png"
            )

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
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                noise_level=noise_level,
                ambient_light_intensity=ambient_light_intensity,
                diffuse_light_intensity=diffuse_light_intensity,
                specular_light_intensity=specular_light_intensity,
                specular_exponent=specular_exponent
            )

            writer.writerow({
                'diameter_mm': diameter_mm,
                'qr_side_length_mm': qr_side_mm,
                'camera_distance_mm': camera_dist_mm,
                'focal_length_mm': focal_length_mm,
                'sphere_rotation_degrees': sphere_rotation_degrees,
                'camera_width_pixels': camera_width_pixels,
                'camera_height_pixels': camera_height_pixels,
                'noise_level': noise_level,
                'ambient_light_intensity': ambient_light_intensity,
                'diffuse_light_intensity': diffuse_light_intensity,
                'specular_light_intensity': specular_light_intensity,
                'specular_exponent': specular_exponent,
                'filename': full_path
            })

print(f"Parameters and filenames recorded in {csv_file}")
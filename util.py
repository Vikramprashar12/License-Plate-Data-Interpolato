import string
import easyocr
import re
import cv2
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5', }

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S', }


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_int_to_char, 4: dict_char_to_int,
               5: dict_char_to_int, 6: dict_char_to_int}
    for j in range(len(text)):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = re.sub(r'[^a-zA-Z0-9]', '', text)

        text = text.upper().replace(' ', '')

        # if license_complies_format(text):
        #     return format_license(text), score
        return text, score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


# def rotate_image_to_align_license_plate(image):
#     """
#     Detects the top or bottom edge of the license plate, calculates the angle,
#     and rotates the image to align the license plate horizontally.

#     Args:
#         image (np.ndarray): Input image (BGR).

#     Returns:
#         np.ndarray: Rotated image.
#     """
#     # Step 1: Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Step 2: Apply edge detection
#     edges = cv2.Canny(gray, 50, 150)
#     # cv2.imshow("Edges", edges)
#     # cv2.waitKey(0)

#     # Step 3: Detect lines using Hough Line Transform
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
#                             threshold=100, minLineLength=50, maxLineGap=10)

#     if lines is not None:
#         # Step 4: Select the longest line as the reference
#         longest_line = max(lines, key=lambda line: np.linalg.norm(
#             (line[0][0] - line[0][2], line[0][1] - line[0][3])))
#         x1, y1, x2, y2 = longest_line[0]

#         # Draw the detected line for debugging
#         debug_image = image.copy()
#         # cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # cv2.imshow("Detected Line", debug_image)
#         # cv2.waitKey(0)

#         # Step 5: Calculate the angle of rotation
#         delta_y = y2 - y1
#         delta_x = x2 - x1
#         angle = np.degrees(np.arctan2(delta_y, delta_x))

#         print(f"Rotation angle: {angle:.2f} degrees")

#         # Step 6: Rotate the image
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)
#         rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

#         # # Show the rotated image
#         # cv2.imshow("Rotated Image", rotated_image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()

#         return rotated_image

#     else:
#         # print("No lines detected!")
#         # cv2.imshow("Original Image", image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         return image


def rotate_image_to_align_license_plate(image):
    """
    Detects the top or bottom edge of the license plate, calculates the angle,
    rotates the image to align the license plate horizontally, and applies noise removal,
    histogram equalization, and adaptive thresholding.

    Args:
        image (np.ndarray): Input image (BGR).

    Returns:
        np.ndarray: Preprocessed and rotated image.
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Step 3: Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        # Step 4: Select the longest line as the reference
        longest_line = max(lines, key=lambda line: np.linalg.norm(
            (line[0][0] - line[0][2], line[0][1] - line[0][3])))
        x1, y1, x2, y2 = longest_line[0]

        # Step 5: Calculate the angle of rotation
        delta_y = y2 - y1
        delta_x = x2 - x1
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Step 6: Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Step 7: Apply preprocessing (noise removal, histogram equalization, adaptive thresholding)
        # Convert the rotated image to grayscale
        rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        # 7.1 Noise removal using Gaussian Blur
        noise_removed = cv2.GaussianBlur(rotated_gray, (5, 5), 0)

        # 7.2 Histogram equalization
        equalized = cv2.equalizeHist(noise_removed)

        # Return the preprocessed image
        return rotated_image

    else:
        # If no lines are detected, return the original grayscale image
        # Apply preprocessing to the original grayscale image
        noise_removed = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(noise_removed)
        return equalized

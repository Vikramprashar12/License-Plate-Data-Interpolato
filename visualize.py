import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y),
             color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y),
             color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1),
             color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y),
             color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


# Load results
results = pd.read_csv('./test_interpolated.csv')

# Load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Downscale resolution for processing
output_width = 1920
output_height = 1080

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use H.264 codec
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (output_width, output_height))

# Process license plates
license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]
                   ['license_number_score'])
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[(results['car_id'] == car_id) &
                                        (results['license_number_score'] == max_)]['license_number'].iloc[0]
    }
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(
        license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Process frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        # Resize frame for processing
        frame = cv2.resize(frame, (output_width, output_height))

        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # Draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace(
                '[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace(
                '[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 255), 12)

            # Crop license plate
            license_crop = license_plate[df_.iloc[row_indx]
                                         ['car_id']]['license_crop']

            H, W, _ = license_crop.shape
            try:
                # Calculate target position for license plate overlay
                start_y = int(car_y1) - H - 100
                end_y = int(car_y1) - 100
                start_x = int((car_x2 + car_x1 - W) / 2)
                end_x = int((car_x2 + car_x1 + W) / 2)

                # Ensure target region dimensions are valid
                if start_x < 0 or end_x > frame.shape[1] or start_y < 0 or end_y > frame.shape[0]:
                    print(
                        f"Skipping license plate overlay for car_id {df_.iloc[row_indx]['car_id']} due to invalid dimensions.")
                    continue

                # Overlay the license crop
                frame[start_y:end_y, start_x:end_x, :] = license_crop

                # Add white background for text
                start_y_text = start_y - 300
                end_y_text = start_y
                if start_y_text >= 0:
                    frame[start_y_text:end_y_text,
                          start_x:end_x, :] = (255, 255, 255)

                # Overlay license plate text
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]
                                  ['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                text_x = int((car_x2 + car_x1 - text_width) / 2)
                text_y = int(start_y - 150 + (text_height / 2))

                if text_x >= 0 and text_x + text_width < frame.shape[1]:
                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]
                                              ['car_id']]['license_plate_number'],
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)

            except Exception as e:
                print(f"Error processing license plate overlay: {e}")

        out.write(frame)

out.release()
cap.release()

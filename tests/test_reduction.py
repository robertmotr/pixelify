import cv2
import os

def find_global_max_min(image_path):
    img = cv2.imread(image_path)

    # image is loaded successfully
    if img is None:
        raise ValueError(f"Error loading image from {image_path}")

    # split the image into RGB channels
    channels = cv2.split(img)

    global_max = [0, 0, 0]
    global_min = [255, 255, 255]

    for channel in channels:
        # maximum and minimum values for each channel
        _, channel_max, _, _ = cv2.minMaxLoc(channel)
        channel_min = cv2.minMaxLoc(channel, None)[0]

        global_max = [max(g_max, c_max) for g_max, c_max in zip(global_max, [channel_max] * 3)]
        global_min = [min(g_min, c_min) for g_min, c_min in zip(global_min, [channel_min] * 3)]

    global_max = [int(g_max) for g_max in global_max]
    global_min = [int(g_min) for g_min in global_min]

    return global_max, global_min

if __name__ == "__main__":
    image_path = os.getcwd() + "/../sample_images/Puzzle_Mountain.png"
    max_values, min_values = find_global_max_min(image_path)

    print("Global max values:", max_values)
    print("Global min values:", min_values)

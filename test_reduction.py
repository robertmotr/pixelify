from PIL import Image

def find_global_max_min(image_path):
    # Open the image
    img = Image.open(image_path)

    # Get pixel data
    pixels = img.getdata()

    # Initialize max and min values for each channel
    global_max = [0, 0, 0, 0]
    global_min = [255, 255, 255, 255]

    # Iterate through all pixels
    for pixel in pixels:
        for i in range(4):  # RGBA channels
            # Update max and min values for each channel
            global_max[i] = max(global_max[i], pixel[i])
            global_min[i] = min(global_min[i], pixel[i])

    return global_max, global_min

if __name__ == "__main__":
    image_path = "path/to/your/image.png"
    max_values, min_values = find_global_max_min(image_path)

    print("Global Max Values (RGBA):", max_values)
    print("Global Min Values (RGBA):", min_values)

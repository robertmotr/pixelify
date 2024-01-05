from PIL import Image

def find_global_max_min(image_path):
    img = Image.open(image_path)

    pixels = img.getdata()

    global_max = [0, 0, 0]
    global_min = [255, 255, 255]

    for pixel in pixels:
        for i in range(3):  # RGB channels
            global_max[i] = max(global_max[i], pixel[i])
            global_min[i] = min(global_min[i], pixel[i])

    return global_max, global_min

if __name__ == "__main__":
    image_path = "../sample_images/Puzzle_Mountain.png"
    max_values, min_values = find_global_max_min(image_path)

    print("Global max values:", max_values)
    print("Global min values:", min_values)

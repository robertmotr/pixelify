def normalize_pixels(input_pixels):
    # Make a copy of the input array to avoid modifying it
    pixels = [pixel.copy() for pixel in input_pixels]

    # Find the minimum and maximum values for each channel
    min_values = [min(pixel[i] for pixel in pixels) for i in range(3)]
    max_values = [max(pixel[i] for pixel in pixels) for i in range(3)]

    # Normalize each respective channel
    for pixel_idx in range(len(pixels)):
        for channel in range(3):
            min_val = min_values[channel]
            max_val = max_values[channel]
            value = pixels[pixel_idx][channel]

            # Prevent division by zero
            if max_val == min_val:
                continue
            
            # Perform normalization
            pixels[pixel_idx][channel] = (int) ((value - min_val) * 255 / (max_val - min_val))

    return pixels

input_pixels = [[1, 2, 3], [4, 5, 6], [7, 8, 9],
                [10, 11, 12], [13, 14, 15], [16, 17, 18],
                [19, 20, 21], [289, 324, 367], [25, 26, 27]]
normalized_pixels = normalize_pixels(input_pixels)

print("Original Pixels:", input_pixels)
print("Normalized Pixels:", normalized_pixels)

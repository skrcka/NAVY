from PIL import Image


def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n


def generate_fractal(xmin, xmax, ymin, ymax, width, height, max_iter):
    img = Image.new('RGB', (width, height))
    pixel_array = img.load()

    for x in range(width):
        for y in range(height):
            zx, zy = x * (xmax - xmin) / (width - 1) + xmin, y * (ymax - ymin) / (height - 1) + ymin
            c = zx + zy * 1j
            z = mandelbrot(c, max_iter)
            r, g, b = z % 8 * 32, z % 16 * 16, z % 32 * 8
            pixel_array[x, y] = (r, g, b)

    return img


if __name__ == '__main__':
    width, height = 3024, 3024
    xmin, xmax = -1, 0.5
    ymin, ymax = -1, 1
    max_iter = 512

    img = generate_fractal(xmin, xmax, ymin, ymax, width, height, max_iter)
    img.show("mandelbrot initial")

def crop(img, f, sx, sy):
    from IPython.display import display
    width, height = img.size

    # Setting the points for cropped image
    left = sx
    top = sy
    right = width * f + left
    bottom = width * f + top

    im1 = img.crop((left, top, right, bottom))
    print(im1.size)
    display(im1)

    return im1
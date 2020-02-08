import numpy as np
import argparse
import struct
from struct import unpack
from collections import deque

import cairocffi as cairo

from quick_redraw.data.metadata_db_session import global_init, create_session
from quick_redraw.etl.store_raw import store_image


def main():
    args = parse_arguments()
    drawing_bin_file = args.drawing_bin_file
    metadata_location = args.metadata_location
    raw_storage_location = args.raw_storage_location
    label = args.label

    init_db(metadata_location=metadata_location)

    drawings_as_raster = load_drawings(drawing_bin_file)

    # Store images
    for drawing in drawings_as_raster:
        store_image(label=label, drawing=drawing, metadata_location=metadata_location,
                    raw_storage_location=raw_storage_location)


def init_db(metadata_location):
    global_init(metadata_location)


def load_drawings(drawing_bin_file):
    output_size = 28  # pixels square
    drawings_as_vector = deque()

    for drawing_as_vector in unpack_drawings(drawing_bin_file):
        drawings_as_vector.append(drawing_as_vector)

    drawings_as_raster = vector_to_raster(drawings_as_vector, side=output_size)
    return drawings_as_raster


def unpack_drawing(file_handle):
    # Source: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))

    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    # Source: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
    """
    padding and line_diameter are relative to the original 256x256 image.

    source: https://github.com/googlecreativelab/quickdraw-dataset/issues/19
    """

    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images


def parse_arguments():
    parser = argparse.ArgumentParser(description="Loads all images from a .bin file to a local or GCP storage")

    parser.add_argument('drawing_bin_file', type=str, action="store",
                        help="Path to .bin file containing drawings to load")
    parser.add_argument('label', type=str, action="store",
                        help="Label for loaded drawings")
    parser.add_argument('metadata_location', type=str, action="store",
                        help="Path to metadata database (local or gcp bucket)")
    parser.add_argument('raw_storage_location', type=str, action="store",
                        help="Location to store raw files (local or gcp bucket)")
    parser.add_argument("--max_images", type=int, action="store", default=100)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

import os


def imitation_arguments(renders=False):
    # by default cuda is used if it is available
    args = {'video_dir': './data/validation/angle1', 'frame_size': (299, 299), 'renders': renders}
    return args

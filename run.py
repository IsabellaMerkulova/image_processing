import os
import texture_segmentation

if __name__ == '__main__':
    result_images = []
    folder = 'images/defects'
    for image in os.listdir(folder):
        # if image.endswith('kp_crop.png'):
        if image.startswith('sample8_'):
            texture_segmentation.get_sliding_window_properties(f'{folder}/{image}')
        # result_images.append(get_sliding_window_properties(f'{folder}/{image}'))
    # show_result_images(result_images)
from videos_to_frames import extract_frames_to_images
from extract_faces import extract_faces
import combine_images
import subprocess

if __name__ == '__main__':
    print('Starting Pre-Processing')
    extract_frames_to_images()
    print('Done extract frames to images')
    subprocess.call(['rm','-r','Dataset'])

    import shutil
    shutil.move('Dataset_frames/Deepfake','Deepfake')
    shutil.move('Dataset_frames/Real', 'Real')
    subprocess.call(['rm','-r','Dataset_frames'])

    extract_faces()
    print('Done extracting faces')
    subprocess.call(['rm','-r','Deepfake','Real'])

    combine_images.copy_images()
    combine_images.combine_images(new_folder = 'Dataset_new')
    print('Done creating final dataaset')
    subprocess.call(['rm','-r','face_images', 'Dataset_final_images'])

    print('splitting dataset')

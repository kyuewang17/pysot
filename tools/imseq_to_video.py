"""
Image Sequence to Video
"""
import cv2
import os
import numpy as np
import glob


siamese_backbone = "siamrpn_alex_dwxcorr_otb"
tracking_results = "results_allframe_exemplar_update_rate_0.0025"
tracking_benchmark = "OTB100"

# Base Path
experiments_path = os.path.join(os.path.dirname(os.getcwd()), "experiments")
base_path = os.path.join(experiments_path, siamese_backbone, tracking_results, tracking_benchmark)


def imseq_to_video(imseq_master_path, image_fmt, result_video_name, result_video_fmt, fourcc="MP4V", frame_rate=15):
    img_array = []
    width, height = -1, -1
    for file_idx, img_file_name in enumerate(sorted(os.listdir(imseq_master_path))):
        # Percentage
        percentage = float(file_idx+1) / len(sorted(os.listdir(imseq_master_path)))*100

        # Store Image Message
        mesg_str = "Appending Image...{%3.3f %s}" % (percentage, chr(37))
        print(mesg_str)

        # Check for File Extension
        _, file_extension = os.path.splitext(os.path.join(imseq_master_path, img_file_name))

        if file_extension.__contains__(image_fmt) is not True:
            assert 0, "Format must be %s...! (current file format is %s)" % (image_fmt, file_extension[1:])

        frame = cv2.imread(os.path.join(imseq_master_path, img_file_name))
        height, width, layers = frame.shape
        img_array.append(frame)
    size = (width, height)

    # Video Save Path
    result_video_name = result_video_name + "." + result_video_fmt
    video_save_path = os.path.join(base_path, result_video_name)

    # Video Writer
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*fourcc), frame_rate, size)

    # Write Images
    for img_array_idx in range(len(img_array)):
        # Percentage
        percentage = (float(img_array_idx+1) / len(img_array))*100

        # Work Message
        mesg_str = "Writing Images...{%3.3f %s}" % (percentage, chr(37))
        print(mesg_str)

        out.write(img_array[img_array_idx])
    out.release()


# Main
def main():
    # Frame Rate
    frame_rate = 15

    sequence_name_list = sorted(os.listdir(os.path.join(base_path, "visualization_results")))

    # For each sequences,
    for seq_idx, sequence_name in enumerate(sequence_name_list):
        # Current Folder Path
        seq_folder_path = os.path.join(base_path, "visualization_results", sequence_name)

        # Set Video Name
        video_name = sequence_name + "__(fps_{0:d})".format(frame_rate)
        video_file_base_path = os.path.join(base_path, "video_results")
        video_file_path = os.path.join(video_file_base_path, video_name)

        # Image Sequence to Video
        imseq_to_video(seq_folder_path, "jpg", video_file_path, "mp4", frame_rate=frame_rate)


# Namespace
if __name__ == '__main__':
    main()

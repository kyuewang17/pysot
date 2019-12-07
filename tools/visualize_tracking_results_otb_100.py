"""
Visualize OTB-100 Results for IITP Final Presentation

"""
import cv2
import numpy as np
import os
import copy


# Visualization Mode
is_visualization = True

# Save Mode
is_save_results = False


def cxywh_to_ltrbxy(bbox_cxywh):
    cx, cy = bbox_cxywh[0], bbox_cxywh[1]
    w, h = bbox_cxywh[2], bbox_cxywh[3]

    ltx, lty = (cx-w/2.0), (cy-h/2.0)
    rbx, rby = (cx+w/2.0), (cy+h/2.0)

    ltrbxy = np.array([ltx, lty, rbx, rby]).reshape(4, 1)

    return ltrbxy


def ltxywh_to_ltrbxy(bbox_ltxywh):
    ltx, lty = bbox_ltxywh[0], bbox_ltxywh[1]
    w, h = bbox_ltxywh[2], bbox_ltxywh[3]

    rbx, rby = ltx+w, lty+h

    ltrbxy = np.array([ltx, lty, rbx, rby]).reshape(4, 1)

    return ltrbxy


def draw_bbox_on_img(target_frame_img, bbox, color):
    """

    :param target_frame_img:
    :param bbox: format: <ltxywh>
    :param color: List of RGB
    :return:
    """
    # Draw Rectangle BBOX
    # bbox_ltrbxy = cxywh_to_ltrbxy(bbox)
    bbox_ltrbxy = ltxywh_to_ltrbxy(bbox)
    draw_bbox_ltrbxy = bbox_ltrbxy.astype(np.int32)
    cv2.rectangle(target_frame_img,
                  (draw_bbox_ltrbxy[0], draw_bbox_ltrbxy[1]), (draw_bbox_ltrbxy[2], draw_bbox_ltrbxy[3]),
                  (color[0], color[1], color[2]), thickness=2)


def draw_fidx_on_img(target_frame_img, fidx, text_color, text_padding, extra_padding=3):
    """

    :param target_frame_img:
    :param fidx:
    :param text_x: <left-top>
    :param text_y: <left-top>
    :param text_color:
    :param text_padding:
    :return:
    """

    # Draw Frame Index Number
    fidx_str = "FIDX:{0:5d}".format(fidx)
    (tw, th) = cv2.getTextSize(fidx_str, cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, thickness=2)[0]
    str_padding_bbox_coords = (
        (int(extra_padding + text_padding / 2.0), int(extra_padding + text_padding / 2.0)),
        (int(extra_padding + tw + text_padding / 2.0), int(extra_padding + th + text_padding / 2.0))
    )
    cv2.rectangle(target_frame_img, str_padding_bbox_coords[0], str_padding_bbox_coords[1],
                  (128, 128, 128), cv2.FILLED)

    # Put Text
    text_lbx, text_lby = int(extra_padding + text_padding / 2.0), int(extra_padding + th + text_padding / 2.0)
    cv2.putText(target_frame_img, fidx_str, (text_lbx, text_lby), cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.2, color=(text_color[0], text_color[1], text_color[2]), thickness=2)


# <User Input> Model Name
model_name = "siamrpn_alex_dwxcorr_otb"
result_folder_name = "results_allframe_exemplar_update_rate_0.0025"

curr_file_base_path = os.getcwd()
experiments_path = os.path.join(os.path.dirname(curr_file_base_path), "experiments")

# Current Model Full Path
model_full_path = os.path.join(experiments_path, model_name)
model_result_dir = os.path.join(model_full_path, result_folder_name)

# FPS Text File (Tentative)
fps_result_txt_file_cands = os.listdir(model_result_dir)
assert (len(fps_result_txt_file_cands) == 2), "Directory Formatting is not right...(Check Directory)"
fps_result_txt_file_path = os.path.join(model_result_dir, fps_result_txt_file_cands[1])

# OTB-100 Tracking Result Text Files
tracking_result_file_base_path = os.path.join(model_result_dir, "OTB100")
tracking_result_file_path = os.path.join(tracking_result_file_base_path, "model")
seq_result_file_name_list = sorted(os.listdir(tracking_result_file_path))
assert (len(seq_result_file_name_list) == 100), "Result TXT File is not appropriate or missing!"

# Generate Result Base Path
qualitative_tracking_result_visualization_path = os.path.join(tracking_result_file_base_path, "visualization_results")
if os.path.isdir(qualitative_tracking_result_visualization_path) is False:
    os.mkdir(qualitative_tracking_result_visualization_path)

# Path to OTB100 Image Sequence
otb100_sequence_base_path = os.path.join(os.path.dirname(curr_file_base_path), "testing_dataset", "OTB100")

# For Each Sequence,
for seq_idx, sequence_file_name in enumerate(seq_result_file_name_list):
    sequence_name = sequence_file_name.split(".")[0]
    img_seq_folder = os.path.join(otb100_sequence_base_path, sequence_name, "img")

    # Read Tracking Result Text File for Current Sequence
    sequence_tracking_result_file_path = os.path.join(tracking_result_file_path, sequence_file_name)
    seq_tracking_results = np.loadtxt(sequence_tracking_result_file_path, delimiter=",", dtype=np.float64)

    # Read Ground Truth Result Text File for Current Sequence
    curr_seq_gt_file_path = os.path.join(otb100_sequence_base_path, sequence_name, "groundtruth.txt")
    seq_gt_results = np.loadtxt(curr_seq_gt_file_path, delimiter=",", dtype=np.float64)

    # Check for Frame Lengths (will select minimum frame length)
    frame_lengths_list = [seq_tracking_results.shape[0], seq_gt_results.shape[0]]

    # Generate Result Base Path to Current Sequence
    curr_seq_result_path = os.path.join(qualitative_tracking_result_visualization_path, sequence_name)
    if os.path.isdir(curr_seq_result_path) is False:
        os.mkdir(curr_seq_result_path)

    # Get Current Sequence Image Frame File Name List
    curr_seq_frame_file_name_list = sorted(os.listdir(img_seq_folder))

    if len(curr_seq_frame_file_name_list) != min(frame_lengths_list):
        print("[WARNING] (Seq:{}) Frame Lengths Does Not Match!....(Results may not be right!)".format(sequence_name))

    # For Each Frames,
    for fidx, frame_file_name in enumerate(curr_seq_frame_file_name_list):
        # Get Current Frame Image File Path
        frame_file_path = os.path.join(img_seq_folder, frame_file_name)

        # Read Image using OpenCV (read in BGR, so needs conversion)
        curr_frame = cv2.cvtColor(cv2.imread(frame_file_path), cv2.COLOR_BGR2RGB)

        # Image for Drawing
        draw_curr_frame = copy.deepcopy(curr_frame)

        # Get Current Frame Index (1)Tracking Result and (2)Ground-Truth Result
        curr_track_bbox, curr_gt_bbox = seq_tracking_results[fidx, :], seq_gt_results[fidx, :]

        # (1) Draw Tracking BBOX (blue bbox)
        draw_bbox_on_img(draw_curr_frame, curr_track_bbox, color=[0, 0, 255])

        # (2) Draw Ground Truth BBOX (red bbox)
        draw_bbox_on_img(draw_curr_frame, curr_gt_bbox, color=[255, 0, 0])

        # Draw Frame Index Count
        draw_fidx_on_img(draw_curr_frame, fidx+1, text_color=[255, 255, 255], text_padding=3)

        # Show Image
        if is_visualization is True:
            window_name = "Sequence:[{0:s}]".format(sequence_name)
            cv2.namedWindow(window_name)
            if fidx == 0:
                cv2.moveWindow(window_name, 400, 200)
            cv2.imshow(window_name, cv2.cvtColor(draw_curr_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            # cv2.destroyWindow(window_name)

        # Save Image
        if is_save_results is True:
            pass


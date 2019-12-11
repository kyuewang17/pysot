from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str,
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.set_defaults(show_video_level=False)


curr_file_base_path = os.getcwd()
experiments_base_path = os.path.join(os.path.dirname(curr_file_base_path), "experiments")
model_base_path = os.path.join(experiments_base_path, "siamrpn_alex_dwxcorr_otb")

# CODE CODE CODE
args_list = []

frame_interval_list = [2, 3, 5, 10, 20]
# frame_interval_list = [20, 50, 75, 100, 150]
interpolation_rate_list = [0.00125, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]

for frame_interval in frame_interval_list:
    for interpolation_rate in interpolation_rate_list:
        curr_args = parser.parse_args()
        curr_args.dataset = "OTB100"
        curr_args.num = 1
        curr_args.tracker_prefix = "model"

        result_folder_name = "results_{0:s}frame_exemplar_update_rate_{1:s}".format(str(frame_interval), str(interpolation_rate))

        curr_args.tracker_path = os.path.join(model_base_path, result_folder_name)

        curr_fps_txt_file_list = glob(os.path.join(curr_args.tracker_path, "*.txt"))
        try:
            assert (len(curr_fps_txt_file_list) == 1), "TXT Files are more than one"
            curr_fps_txt_file_path = curr_fps_txt_file_list[0]
            curr_fps_txt_file_name = curr_fps_txt_file_path.split("/")[-1]
            curr_fps_str = curr_fps_txt_file_name.split("__")[1].split("[")[1].split("]")[0]
            fps_mesg = "update_{0:s}_{1:s} ==>> [Mean FPS: {2:s}]".format(str(frame_interval), str(interpolation_rate), curr_fps_str)
        except:
            fps_mesg = "update_{0:s}_{1:s} ==>> [Mean FPS: NAN]".format(str(frame_interval), str(interpolation_rate))

        print(fps_mesg)

        curr_args.ablation_name = result_folder_name

        args_list.append(curr_args)
        # del curr_args

pass

######### Manually Add Parser Arguments for Debugging Code #########
def main(args):
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]
    # trackers = os.listdir(os.path.join(args.tracker_path, args.dataset, args.tracker_prefix))

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                            '../testing_dataset'))
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)


#


if __name__ == '__main__':
    for args_idx, args in enumerate(args_list):
        ablation_mesg = "({0:s}/{1:s}) Current Ablation: [{2:s}]".format(str(args_idx+1), str(len(args_list)), args.ablation_name)
        print(ablation_mesg)
        main(args)
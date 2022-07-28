import argparse
import os

from sequentialign import *
from data_preparation import *
from result_handler import *

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dataset_name', type=str, default='sample', help='sample or other dataset name')
    parser.add_argument('-ev', '--eval_only', type=str, default='yes', help='yes=evaluate metrics from cached results, no=run algorithms and overwrite cached results')
    parser.add_argument('-th', '--filter_thresholds', nargs='+', type=int, default=[25, 33, 50], help='thresholds for the sequentialign irrelevant content filter, input one or more integers between 0 and 100')
    return parser.parse_args()

def main(args):

    dn = args.dataset_name

    path = "./data/" + args.dataset_name + "/"
    subd_count = count_subd(path)

    filename = args.dataset_name

    raw_dumper = ResultDump(filename)
    thresholds = args.filter_thresholds

    if not args.eval_only == "yes":
        for deck_no in range(1, subd_count + 1):
            thold = 0
            print("processing deck", deck_no)
            s_vlocal, c_vlocal, s_clocal, c_clocal, s_relocal, c_relocal, gs_timings = or_time_grid_terms(args.dataset_name, deck_no, "fixed", 30)

            raw_dumper.save_gs_timings(deck_no, gs_timings)
            raw_dumper.save_sr(deck_no, s_relocal)
            raw_dumper.save_sra(deck_no, thold, s_relocal)
            raw_dumper.save_cr(deck_no, thold, c_relocal)

            as1, av1 = bifilter_redundancies(s_vlocal, c_vlocal, s_relocal, c_relocal, 3, gs_timings, 5, thresholds)
            as2, av2 = bifilter_redundancies(s_vlocal, c_vlocal, s_relocal, c_relocal, 4, gs_timings, 4, thresholds)
            as3, av3 = bifilter_redundancies(s_vlocal, c_vlocal, s_relocal, c_relocal, 5, gs_timings, 3, thresholds)
            slide_irrels, video_irrels = bifilter_getirrels(as1, as2, as3, av1, av2, av3, thresholds)

            for thold in thresholds:
                irrel_slide = slide_irrels[thold]
                irrel_video = av1[thold]
                new_sv, new_sc, new_sr, new_cv, new_cc, new_cr = get_filtered_data(irrel_video, irrel_slide, s_vlocal,
                                                                                   s_clocal, s_relocal, c_vlocal, c_clocal,
                                                                                   c_relocal)

                raw_dumper.save_cr(deck_no, thold, new_cr)
                raw_dumper.save_sra(deck_no, thold, new_sr)

                dtw_timings, dist = sequentialign_align(new_sv, new_cv, new_sr, new_cr)
                raw_dumper.save_results(deck_no, thold, "seqalign", dtw_timings)

        raw_dumper.dump()

    raw_dumper_2 = ResultDump(filename)
    raw_dumper_2.load()

    sqalign_kcrh = KeyedCategoricalResultHandler("Sequentialign")

    for deck_no in range(1, subd_count + 1):
        s_relocal = raw_dumper_2.get_sr(deck_no)

        gs_timings = get_gs_timings(dn, deck_no)

        for thold in thresholds:
            new_cr = raw_dumper_2.get_cr(deck_no, thold)

            dtw_timings = raw_dumper_2.get_results(deck_no, thold, "seqalign")
            oiou_excl = get_miou(gs_timings, dtw_timings, s_relocal)
            vacc = utp_video_accuracy(gs_timings, dtw_timings, new_cr, dn, deck_no)
            sqalign_kcrh.add_results(vacc, oiou_excl, thold)

    for thold in thresholds:
        sqalign_kcrh.print_avgs(thold)

def count_subd(path):
    count = 0
    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            count+=1
    return count

if __name__ == '__main__':
    main(parse_arguments())
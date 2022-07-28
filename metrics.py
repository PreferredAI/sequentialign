from pptx import Presentation
import numpy as np

def get_slide_accuracy(gs_timings, dtw_timings, dn, deck_no, slide_rel):
    pptx_path = "./data/" + dn + "/" + str(deck_no) + "/slides.pptx"
    prs = Presentation(pptx_path)
    num_slides = len(prs.slides)

    variants = [15, 120]

    score = {}
    score_iou = {}
    score_iou_excl = {}
    sie_count = {}
    for v in variants:
        score[v] = 0
        score_iou[v] = 0
        score_iou_excl[v] = 0
        sie_count[v] = 0
        for key in gs_timings:
            if key in dtw_timings:
                dtw_time = dtw_timings[key][0]
                acceptable_range = gs_timings[key]
                if dtw_time >= (acceptable_range[0] - v) and dtw_time <= (acceptable_range[0] + v):
                    score[v] += 1
                    itn = overlap(dtw_timings[key][0], dtw_timings[key][1], gs_timings[key][0], gs_timings[key][1])
                    unn = union(dtw_timings[key][0], dtw_timings[key][1], gs_timings[key][0], gs_timings[key][1])
                    iou = itn / unn
                    score_iou[v] += iou
                    score_iou_excl[v] += iou
                    sie_count[v] += 1

    for key in slide_rel.values():
        if key not in dtw_timings:
            if key not in gs_timings:
                for v in variants:
                    score[v] += 1
                    score_iou[v] += 1

    # print(score[120], score_iou[120], score_iou_excl[120], sie_count[120])
    for v in variants:
        if score[v] != 0:
            score_iou[v] = round((score_iou[v] / score[v]), 3)
        if sie_count[v] != 0:
            score_iou_excl[v] = round((score_iou_excl[v] / sie_count[v]), 3)
        score[v] = round((score[v] / len(slide_rel.values()) * 100), 3)

    return score[120]

def get_miou(gs_timings, dtw_timings, slide_rel):
    score = 0.0
    pos_score = 0
    true_neg = 0
    count = 0

    for key in dtw_timings:
        if key in gs_timings:
            itn = overlap(dtw_timings[key][0], dtw_timings[key][1], gs_timings[key][0], gs_timings[key][1])
            if itn > 0:
                pos_score += 1
            unn = union(dtw_timings[key][0], dtw_timings[key][1], gs_timings[key][0], gs_timings[key][1])
            score += itn / unn
            count += 1

    # print(pos_score)
    for key in slide_rel.values():
        if key not in dtw_timings:
            if key not in gs_timings:
                true_neg += 1
                count += 1

    return (score+true_neg) / len(slide_rel.values())

def utp_video_accuracy(gs_timings, dtw_timings, chunk_rel, dn, deck_no):
    groundstamps = np.loadtxt("./data/" + dn + "/" + str(deck_no) + "/annotation.txt", dtype="str")
    duration = int(groundstamps[0])
    relevant_seconds = np.zeros(shape=(duration))
    binary_correctness = np.zeros(shape=(duration))
    num_slides_linked = np.zeros(shape=(duration))
    accessible_seconds = np.zeros(shape=(duration))
    score = 0.0

    for key in chunk_rel:
        for i in range(int(chunk_rel[key][0]), int(chunk_rel[key][1])):
            if i < duration:
                accessible_seconds[i] = 1

    for i in range(1, duration+1):
        for key in gs_timings:
            if i >= gs_timings[key][0] and i < gs_timings[key][1]:
                relevant_seconds[i-1] = key
    for key in dtw_timings:
        for i in range(int(dtw_timings[key][0]), int(dtw_timings[key][1])):
            if i < duration:
                if accessible_seconds[i-1] == 1:
                    num_slides_linked[i - 1] += 1
                    if relevant_seconds[i-1] == key:
                        binary_correctness[i-1] = 1
    for i in range(len(relevant_seconds)):
        if relevant_seconds[i] == 0:
            if num_slides_linked[i] == 0:
                score += 1
        else:
            if binary_correctness[i] == 1:
                score += 1

    return score / duration

def overlap(start1, end1, start2, end2):
    return max(max((end2 - start1), 0) - max((end2 - end1), 0) - max((start2 - start1), 0), 0)

def union(start1, end1, start2, end2):
    if overlap(start1, end1, start2, end2) > 0:
        return max(end1, end2) - min(start1, start2)
    else:
        return (end1 - start1 + end2 - start2)
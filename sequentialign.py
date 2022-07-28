from scipy.spatial.distance import euclidean, cosine
import scipy
import numpy as np
import random

from metrics import *

INF = 1.7976931348623157e+308

"""
Basic DTW
"""

def dtw_basic(data, query):
    compute_matrix = np.zeros(shape=(len(data) + 1, len(query) + 1))
    for i in range(len(data) + 1):
        for j in range(len(query) + 1):
            compute_matrix[i][j] = INF

    compute_matrix[0][0] = 0

    for i in range(1, len(data) + 1):
        for j in range(1, len(query) + 1):
            cost = cosine(data[i - 1], query[j - 1])
            compute_matrix[i][j] = cost + min(compute_matrix[i - 1][j], compute_matrix[i][j - 1],
                                              compute_matrix[i - 1][j - 1])

    return dtw_traceback(compute_matrix), compute_matrix[len(data), len(query)]

def dtw_traceback(compute_matrix):
    i, j = compute_matrix.shape
    i -= 1
    j -= 1
    path = dict()
    path[j] = i
    while i != 0 and j != 0:
        if compute_matrix[i - 1][j] < compute_matrix[i][j - 1]:
            if compute_matrix[i - 1][j] < compute_matrix[i - 1][j - 1]:
                i -= 1
            else:
                i -= 1
                j -= 1
                if j != 0:
                    path[j] = i
        else:
            if compute_matrix[i][j - 1] < compute_matrix[i - 1][j - 1]:
                j -= 1
                if j != 0:
                    path[j] = i
            else:
                i -= 1
                j -= 1
                if j != 0:
                    path[j] = i

    return path

def bifilter_redundancies(s_vlocal, c_vlocal, s_relocal, c_relocal, rv, gs_timings, multiplier_max, thlds):
    slide_scores = np.zeros(shape=(len(s_vlocal)))
    video_scores = np.zeros(shape=(len(c_vlocal)))
    amalgamated_slides = {}
    amalgamated_videos = {}
    for qw in range(rv * 2, rv * multiplier_max):
        rel_slide_count, rel_chunk_count = bifilter_subroutine(s_vlocal, c_vlocal, rv, qw, 1)
        for i in range(len(slide_scores)):
            slide_scores[i] += rel_slide_count[i]
        for i in range(len(video_scores)):
            video_scores[i] += rel_chunk_count[i]

    globaslide = []
    globavideo = []
    for j in range(len(c_vlocal)):
        globavideo.append(video_scores[j])
    for j in range(len(s_vlocal)):
        globaslide.append(slide_scores[j])

    for thld in thlds:
        slide_thld = np.percentile(globaslide, thld)
        video_thld = np.percentile(globavideo, thld)

        slide_irrels = set()
        for i in range(len(slide_scores)):
            if slide_scores[i] < slide_thld:
                slide_irrels.add(i)
        amalgamated_slides[thld] = slide_irrels

        video_irrels = set()
        for i in range(len(video_scores)):
            if video_scores[i] < video_thld:
                video_irrels.add(i)
        amalgamated_videos[thld] = video_irrels

    return amalgamated_slides, amalgamated_videos

def bifilter_subroutine(slide_vectors, chunk_vectors, qw, r, thld):
    rel_chunk_count = np.zeros(shape=(len(chunk_vectors)))
    rel_slide_count = np.zeros(shape=(len(slide_vectors)))
    for j in range(len(slide_vectors) - qw):
        loc, cost, path, measures = query_subroutine_measured_es(chunk_vectors, slide_vectors[j:j + qw], r)
        for k in range(loc, loc + r):
            if measures[(k - loc)] < thld:
                score = 100 / cost
                if measures[(k-loc)] > 0:
                    score = score / measures[(k - loc)]
                rel_chunk_count[k] += score

        occcount = list(path.values())
        for i in path:
            if (i+1) in path:
                endet = path[i+1]
            else:
                endet = r+1
            for k in range(path[i], endet):
                score = 100
                if measures[k-1] > 0:
                    score = score / cost / measures[k-1]
                if occcount.count(path[i]) > 1:
                    score = score / occcount.count(path[i])
                rel_slide_count[j] += score

    return rel_slide_count, rel_chunk_count

def bifilter_getirrels(as1, as2, as3, av1, av2, av3, thlds):
    slide_irrels = {}
    video_irrels = {}

    for thld in thlds:
        s1 = as1[thld]
        s2 = as2[thld]
        s3 = as3[thld]
        intersection = set.intersection(s1, s2, s3)
        slide_irrels[thld] = intersection

        v1 = av1[thld]
        v2 = av2[thld]
        v3 = av3[thld]
        intersection = set.intersection(v1, v2, v3)
        video_irrels[thld] = intersection

    return slide_irrels, video_irrels

def query_subroutine_measured_es(data, query, r):
    bsf = INF
    loc = INF
    correct_path = []
    correct_measures = np.ones(shape=(r))
    for i in range(len(data) - r):
        path, cosine_measures, dist = filtering_dtw_measured_es(data[i:i+r], query, bsf)
        if not path is None:
            if dist < bsf:
                bsf = dist
                loc = i
                correct_path = path
                correct_measures = cosine_measures
    return loc, bsf, correct_path, correct_measures

def filtering_dtw_measured_es(data, query, bsf):

    compute_matrix = np.zeros(shape=(len(data) + 1, len(query) + 1))
    for i in range(len(data) + 1):
        for j in range(len(query) + 1):
            compute_matrix[i][j] = INF

    compute_matrix[0][0] = 0

    for i in range(1, len(data) + 1):
        for j in range(1, len(query) + 1):
            cost = cosine(data[i - 1], query[j - 1])
            compute_matrix[i][j] = cost + min(compute_matrix[i - 1][j], compute_matrix[i][j - 1],
                                              compute_matrix[i - 1][j - 1])
        if compute_matrix[i][len(query)] > bsf:
            return None, None, None

    path, cosine_measures = filtering_traceback_measured(compute_matrix)

    return path, cosine_measures, compute_matrix[len(data), len(query)]

def filtering_traceback_measured(compute_matrix):
    i, j = compute_matrix.shape
    i -= 1
    j -= 1
    cosine_measures = np.ones(shape=(i+1))
    path = dict()
    while i > 0 and j > 0:
        hoz = compute_matrix[i][j] - compute_matrix[i][j - 1]
        vert = compute_matrix[i][j] - compute_matrix[i - 1][j]
        diag = compute_matrix[i][j] - compute_matrix[i - 1][j - 1]

        vals = [hoz, vert, diag]
        max_diff, max_idx = max(vals), vals.index(max(vals))

        if max_idx == 0:
            path[j] = i
            j -= 1
            cosine_measures[i] = max_diff
        if max_idx == 1:
            path[j] = i
            i -= 1
            cosine_measures[i] = max_diff
        if max_idx == 2:
            path[j] = i
            i -= 1
            j -= 1
            cosine_measures[i] = max_diff

    return path, cosine_measures

def get_filtered_data(irrel_videos, irrel_slides, s_vlocal, s_clocal, s_relocal, c_vlocal, c_clocal, c_relocal):
    new_sv, new_sc, new_sr = [], [], {}

    for i in range(len(s_vlocal)):
        if not i in irrel_slides:
            new_sr[len(new_sv)] = s_relocal[i]
            new_sv.append(s_vlocal[i])
            new_sc.append(s_clocal[i])

    new_cv, new_cc, new_cr = [], [], {}

    for i in range(len(c_vlocal)):
        if not i in irrel_videos:
            new_cr[len(new_cv)] = c_relocal[i]
            new_cv.append(c_vlocal[i])
            new_cc.append(c_clocal[i])

    return new_sv, new_sc, new_sr, new_cv, new_cc, new_cr

def sequentialign_align(slide_vectors, chunk_vectors, slide_rel, time_rel):
    lowest_gf = INF
    lowest_dist = INF
    for GRID_FACTOR in range(1, len(slide_vectors)):
        chunk_grid, slide_grid, interim_result_grid = split_grid(chunk_vectors, slide_vectors, GRID_FACTOR)
        avg_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))
        # print(type(avg_result_grid), avg_result_grid.shape)

        for i in range(len(chunk_grid)):
            cell_chunks = chunk_vectors[chunk_grid[i][0]:chunk_grid[i][1] + 1]
            # print("INITCELLS: ", chunk_grid)
            for j in range(len(slide_grid)):
                cell_slides = slide_vectors[slide_grid[j][0]:slide_grid[j][1]+1]
                path, dist = dtw_basic(cell_chunks, cell_slides)
                interim_result_grid[i][j] = dist
                avg_result_grid[i][j] = dist / len(cell_chunks)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(avg_result_grid)

        for i in range(len(col_ind)-1):
            comb_slides = []
            comb_chunks = []
            # print(slide_grid[col_ind[i]], slide_grid[col_ind[i+1]], chunk_grid[row_ind[i]], chunk_grid[row_ind[i+1]])
            for j in range(int(slide_grid[col_ind[i]][0]), int(slide_grid[col_ind[i]][1])+1):
                comb_slides.append(j)
            for j in range(int(slide_grid[col_ind[i+1]][0]), int(slide_grid[col_ind[i+1]][1])+1):
                comb_slides.append(j)
            for j in range(int(chunk_grid[row_ind[i]][0]), int(chunk_grid[row_ind[i]][1]) + 1):
                comb_chunks.append(j)
            for j in range(int(chunk_grid[row_ind[i + 1]][0]), int(chunk_grid[row_ind[i + 1]][1]) + 1):
                comb_chunks.append(j)

            # print(comb_chunks, comb_slides)
            comb_cell_slides = [slide_vectors[j] for j in comb_slides]
            comb_cell_chunks = [chunk_vectors[j] for j in comb_chunks]
            path, dist = dtw_basic(comb_cell_chunks, comb_cell_slides)

            cell_one_slides = []
            cell_two_slides = []
            for j in range(1, slide_grid[col_ind[i]][1] - slide_grid[col_ind[i]][0] + 2):
                cell_one_slides.append(j)
            for j in range(slide_grid[col_ind[i]][1] - slide_grid[col_ind[i]][0] + 2, len(comb_cell_slides)+1):
                cell_two_slides.append(j)

            cell_one_range = (0, path[cell_one_slides[len(cell_one_slides)-1]+1]-2)
            cell_two_range = (path[cell_two_slides[0]]-1, len(comb_cell_chunks)-1)
            # print(cell_one_range, cell_two_range, len(chunk_vectors))
            chunk_grid[row_ind[i]] = (comb_chunks[cell_one_range[0]], comb_chunks[cell_one_range[1]])
            chunk_grid[row_ind[i+1]] = (comb_chunks[cell_two_range[0]], comb_chunks[cell_two_range[1]])

        interim_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))
        avg_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))

        # print(type(avg_result_grid), avg_result_grid.shape)

        for i in range(len(chunk_grid)):
            cell_chunks = chunk_vectors[chunk_grid[i][0]:chunk_grid[i][1] + 1]
            # print("CELLS: ", chunk_grid)
            for j in range(len(slide_grid)):
                cell_slides = slide_vectors[slide_grid[j][0]:slide_grid[j][1] + 1]
                path, dist = dtw_basic(cell_chunks, cell_slides)
                interim_result_grid[i][j] = dist
                avg_result_grid[i][j] = dist / len(cell_chunks)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(avg_result_grid)

        dtw_dist = interim_result_grid[row_ind, col_ind].sum()
        # print(GRID_FACTOR, dtw_dist)
        if dtw_dist < lowest_dist:
            lowest_dist = dtw_dist
            lowest_gf = GRID_FACTOR

    GRID_FACTOR = lowest_gf
    # print("Lowest GF: ", GRID_FACTOR)

    chunk_grid, slide_grid, interim_result_grid = split_grid(chunk_vectors, slide_vectors, GRID_FACTOR)

    for cycle_count in range(0, 1):
        path_matrix = [[None for i in range(GRID_FACTOR)] for j in range(GRID_FACTOR)]
        interim_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))
        avg_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))

        for i in range(len(chunk_grid)):
            cell_chunks = chunk_vectors[chunk_grid[i][0]:chunk_grid[i][1] + 1]
            for j in range(len(slide_grid)):
                cell_slides = slide_vectors[slide_grid[j][0]:slide_grid[j][1]+1]
                path, dist = dtw_basic(cell_chunks, cell_slides)
                path_matrix[i][j] = path
                interim_result_grid[i][j] = dist
                avg_result_grid[i][j] = dist / len(cell_chunks)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(avg_result_grid)
        # print(row_ind, col_ind)
        dtw_dist = interim_result_grid[row_ind, col_ind].sum()

        for i in range(len(col_ind)-1):
            comb_slides = []
            comb_chunks = []
            # print(slide_grid[col_ind[i]], slide_grid[col_ind[i+1]], chunk_grid[row_ind[i]], chunk_grid[row_ind[i+1]])
            for j in range(int(slide_grid[col_ind[i]][0]), int(slide_grid[col_ind[i]][1])+1):
                comb_slides.append(j)
            for j in range(int(slide_grid[col_ind[i+1]][0]), int(slide_grid[col_ind[i+1]][1])+1):
                comb_slides.append(j)
            for j in range(int(chunk_grid[row_ind[i]][0]), int(chunk_grid[row_ind[i]][1]) + 1):
                comb_chunks.append(j)
            for j in range(int(chunk_grid[row_ind[i + 1]][0]), int(chunk_grid[row_ind[i + 1]][1]) + 1):
                comb_chunks.append(j)

            # print(comb_chunks, comb_slides)
            comb_cell_slides = [slide_vectors[j] for j in comb_slides]
            comb_cell_chunks = [chunk_vectors[j] for j in comb_chunks]
            path, dist = dtw_basic(comb_cell_chunks, comb_cell_slides)
            # print(i, path, dist)

            cell_one_slides = []
            cell_two_slides = []
            for j in range(1, slide_grid[col_ind[i]][1] - slide_grid[col_ind[i]][0] + 2):
                cell_one_slides.append(j)
            for j in range(slide_grid[col_ind[i]][1] - slide_grid[col_ind[i]][0] + 2, len(comb_cell_slides)+1):
                cell_two_slides.append(j)

            cell_one_range = (0, path[cell_one_slides[len(cell_one_slides)-1]+1]-2)
            cell_two_range = (path[cell_two_slides[0]]-1, len(comb_cell_chunks)-1)
            # print(cell_one_range, cell_two_range)
            chunk_grid[row_ind[i]] = (comb_chunks[cell_one_range[0]], comb_chunks[cell_one_range[1]])
            chunk_grid[row_ind[i+1]] = (comb_chunks[cell_two_range[0]], comb_chunks[cell_two_range[1]])
            # print(chunk_grid)

    path_matrix = [[None for i in range(GRID_FACTOR)] for j in range(GRID_FACTOR)]
    avg_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))

    for i in range(len(chunk_grid)):
        cell_chunks = chunk_vectors[chunk_grid[i][0]:chunk_grid[i][1] + 1]
        for j in range(len(slide_grid)):
            cell_slides = slide_vectors[slide_grid[j][0]:slide_grid[j][1] + 1]
            path, dist = dtw_basic(cell_chunks, cell_slides)
            path_matrix[i][j] = path
            interim_result_grid[i][j] = dist
            avg_result_grid[i][j] = dist / len(cell_chunks)

    # print(row_ind, col_ind)
    dtw_dist = interim_result_grid[row_ind, col_ind].sum()

    dtw_timings = dict()
    for i in range(len(col_ind)):
        slide_grid_idx = col_ind[i]
        chunk_grid_idx = row_ind[i]
        slide_grid_range = slide_grid[slide_grid_idx]
        slide_grid_start = slide_grid[slide_grid_idx][0]
        chunk_grid_start = chunk_grid[chunk_grid_idx][0]
        path = path_matrix[chunk_grid_idx][slide_grid_idx]
        path = dict(sorted(path.items()))
        try:
            for j in path:
                if j == 1:
                    dtw_timings[slide_rel[slide_grid_start]] = (
                    time_rel[chunk_grid_start][0], time_rel[chunk_grid_start + path[j] - 1][1])
                else:
                    if path[j] == path[j - 1]:
                        interval = (time_rel[chunk_grid_start][1] - time_rel[chunk_grid_start][0]) // 2
                        prev_st = dtw_timings[slide_rel[slide_grid_start + j - 2]][0]
                        prev_et = dtw_timings[slide_rel[slide_grid_start + j - 2]][1]
                        dtw_timings[slide_rel[slide_grid_start + j - 2]] = (prev_st, max(prev_st, prev_et - interval))
                        dtw_timings[slide_rel[slide_grid_start + j - 1]] = (prev_et - interval, prev_et)
                    else:
                        dtw_timings[slide_rel[slide_grid_start + j - 1]] = (
                        time_rel[chunk_grid_start + path[j - 1]][0], time_rel[chunk_grid_start + path[j] - 1][1])
        except:
            print(j)
            print(slide_grid[slide_grid_idx], chunk_grid[chunk_grid_idx], slide_rel[slide_grid_start],
                  slide_rel[slide_grid_range[1]], time_rel[chunk_grid_start], time_rel[chunk_grid[chunk_grid_idx][1]],
                  path)
            raise
    return dtw_timings, dtw_dist

def split_grid(chunk_vectors, slide_vectors, grid_factor):
    chunk_range = np.array_split(range(len(chunk_vectors)), grid_factor)
    slide_range = np.array_split(range(len(slide_vectors)), grid_factor)

    # print(chunk_range, "\nS:", slide_range)
    chunk_grid = []
    slide_grid = []

    for range_arr in chunk_range:
        array_length = len(range_arr)
        first_element = range_arr[0]
        last_element = range_arr[array_length - 1]
        chunk_grid.append((first_element, last_element))

    for range_arr in slide_range:
        array_length = len(range_arr)
        first_element = range_arr[0]
        last_element = range_arr[array_length - 1]
        slide_grid.append((first_element, last_element))

    interim_result_grid = np.zeros(shape=(len(slide_grid), len(chunk_grid)))

    return chunk_grid, slide_grid, interim_result_grid

def random_align(s_vlocal, c_vlocal, s_relocal, c_relocal):
    random_timings = {}
    chunk_grid, slide_grid, _ = split_grid(c_vlocal, s_vlocal, len(s_vlocal))
    lumps = list(range(len(s_vlocal)))
    random.shuffle(lumps)
    for i in range(len(lumps)):
        random_timings[s_relocal[i]] = (c_relocal[chunk_grid[lumps[i]][0]][0], c_relocal[chunk_grid[lumps[i]][1]][1])
    return random_timings
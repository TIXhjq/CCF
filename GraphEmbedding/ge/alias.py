import numpy as np


def create_alias_table(area_ratio):
    all_probability=area_ratio
    num_probability = len(all_probability)

    all_probability = list((np.array(all_probability) * num_probability) / np.sum(all_probability))

    small, large = [], []
    prab, alias = [-1] * num_probability, [-1] * num_probability

    format_count = 0
    for prob_rank in range(num_probability):
        if all_probability[prob_rank] == 1:
            prab[prob_rank] = 1
            alias[prob_rank] = -1
            format_count += 1
        elif all_probability[prob_rank] > 1:
            large.append(prob_rank)
        else:
            small.append(prob_rank)

    if format_count == num_probability:
        return prab, alias

    while 1:
        if len(small) == 0:
            break
        if len(large) == 0:
            break
        small_rank = small.pop()
        small_data = all_probability[small_rank]
        need_data = 1 - small_data
        large_rank = large.pop()
        rest_data = all_probability[large_rank] - need_data

        prab[small_rank] = small_data
        alias[small_rank] = large_rank
        all_probability[large_rank] = rest_data

        if rest_data == 1:
            prab[large_rank] = 1
            alias[large_rank] = -1

        elif rest_data > 1:
            large.append(large_rank)
        else:
            small.append(large_rank)

    while len(small) != 0:
        small_rank = small.pop()
        prab[small_rank] = 1
    while len(large) != 0:
        large_rank = large.pop()
        prab[large_rank] = 1

    return prab, alias
    # """
    #
    # :param area_ratio: sum(area_ratio)=1
    # :return: accept,alias
    # """
    # l = len(area_ratio)
    # accept, alias = [0] * l, [0] * l
    # small, large = [], []
    # area_ratio_ = np.array(area_ratio) * l
    # for i, prob in enumerate(area_ratio_):
    #     if prob < 1.0:
    #         small.append(i)
    #     else:
    #         large.append(i)
    #
    # while small and large:
    #     small_idx, large_idx = small.pop(), large.pop()
    #     accept[small_idx] = area_ratio_[small_idx]
    #     alias[small_idx] = large_idx
    #     area_ratio_[large_idx] = area_ratio_[large_idx] - \
    #         (1 - area_ratio_[small_idx])
    #     if area_ratio_[large_idx] < 1.0:
    #         small.append(large_idx)
    #     else:
    #         large.append(large_idx)
    #
    # while large:
    #     large_idx = large.pop()
    #     accept[large_idx] = 1
    # while small:
    #     small_idx = small.pop()
    #     accept[small_idx] = 1
    #
    # return accept, alias

from numpy import random

def alias_sample(accept, alias,rank=None):
    """

    :param accept:
    :param alias:
    :return: sample index
    """

    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

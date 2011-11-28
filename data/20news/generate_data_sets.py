#!/usr/bin/env python

"""
Generate pair-wise datasets for 20 news groups
Author: Chen Liu
"""

import sys
import math
from os.path import join as path_join

def split_data_sets(pos_class_list, neg_class_list, input_folder,
        output_folder, exclude_id_file):
    """
    Split the data sets, where every class in pos_class_list will be labeled 1,
    every class in neg_class_list will be labeled -1.
    Three types of data will be generated:
     1 frequency: train-freq.txt, test-freq.txt
     2 tfidf: train-tfidf.txt, test-tfidf.txt
     3 naive bayesian features: train-nb.txt, test-nb.txt
    """
    

    #load stop word list
    stop_words = set()
    swf = open(exclude_id_file, 'r')
    lines = swf.readlines()
    for line in lines:
        stop_words.add(int(line.strip()))
    swf.close()
    lines = None

    for phase in ["test", "train"]:
        #define required data structures
        global_tf = {}
        freq_matrix = {}
        df = {}
        #load the labels
        labelf = open(path_join(input_folder, "%s.label" % phase), "r")
        labels = [int(s.strip()) for s in labelf.readlines()]
        labelf.close()
        #load the data
        dataf = open(path_join(input_folder, "%s.data" % phase), "r")
        line = dataf.readline()
        while(line != ""):
            did, tid, freq = [int(i) for i in line.split()]
            label = labels[did - 1]
            if((label in pos_class_list 
                or label in neg_class_list) and (
                    not tid in stop_words)):
                df[tid] = df.get(tid, 0) + 1
                #global_tf[tid] = global_tf.get(tid, 0) + freq
                freq_matrix[did] = freq_matrix.get(did, {})
                freq_matrix[did][tid] = freq
            line = dataf.readline()
        dataf.close()
        nd = len(freq_matrix)
        o_freq_f = open(path_join(output_folder, "%s-freq.txt" % phase), "w")
        o_tfidf_f = open(path_join(output_folder, "%s-tfidf.txt" % phase), "w")
        for did, freq_map in freq_matrix.iteritems():
            if (labels[did - 1] in pos_class_list):
                label = 1
            elif(labels[did - 1] in neg_class_list):
                label = -1
            for tid, freq in freq_map.iteritems():
                o_freq_f.write("%d,%d " % (tid, freq))
                idf = math.log(nd / df[tid])
                o_tfidf_f.write("%d,%f " % (tid, freq * idf))
            o_freq_f.write("-1,%d\n" % label)
            o_tfidf_f.write("-1,%d\n" % label)
        o_freq_f.close()
        o_tfidf_f.close()

def get_stop_words_list(stop_word_file, vocabulary_file, exclude_id_file):
    stop_words = set()
    swf = open(stop_word_file, 'r')
    line = swf.readline()
    while(line != ''):
        line = line.strip()
        if(line != ""):
            stop_words.add(line)
        line = swf.readline()
    swf.close()
    print len(stop_words)
    vf = open(vocabulary_file, 'r')
    eif = open(exclude_id_file, 'w')
    line = vf.readline()
    count = 1
    while(line != ''):
        line = line.strip()
        if(line in stop_words):
            eif.write('%d\n' % count)
        count += 1
        line = vf.readline()
    eif.close()
    vf.close()



if __name__ == '__main__':
    #split_data_sets([2,3,4,5,6], [8,9,10,11], ".", "comp-rec", "exclude-words.txt")
    #split_data_sets([4], [5], ".", "pc-mac", "exclude-words.txt")
    split_data_sets([17,18,19], [1,16,20], ".", "politic-religion",
            "exclude-words.txt")

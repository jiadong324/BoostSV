#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jiadong Lin

@contact: jdlin@uw.edu

@time: 12/21/24
'''
import logging

import pandas as pd
import re
from spoa import poa
import numpy as np
import math
import joblib
from edlib import align
import fastcluster


from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster

# from Settings import depth_bed


def predict_main(svs_tbl, feat_df, model_path):

    features = ['clusters_ent', 'close_pos_p', 'close_len_p', 'close_l_iden', 'close_r_iden', 'ref_code', 'plat_code']
    # features = ['clusters_ent', 'close_pos_p', 'close_len_p', 'close_l_iden', 'close_r_iden']

    classifier = joblib.load(model_path)

    X = feat_df[features].to_numpy()
    y_pred_prob = classifier.predict_proba(X)
    # y_pred = classifier.predict(X_test)

    feat_df['prob'] = [ele[1] for ele in y_pred_prob]
    feat_df['qual'] = feat_df.apply(assign_weighted_qual, axis=1)
    
    feat_df = feat_df.set_index('svid')
    headers = list(svs_tbl.columns) + ['SIG_READS_NUM','PRED_PROB','QUAL']

    ## Save prediction output
    outputs = []
    sv_tracker = set()
    for idx, row in svs_tbl.iterrows():
        svid = row['ID']
        if svid in sv_tracker:
            logging.warning(f'Duplicated SV: {svid}')
        sv_tracker.add(svid)
        if svid not in feat_df.index:
            sv_info = row.tolist() + ['NA', 'NA', 'NA']
            outputs.append(sv_info)
            continue
        prob = feat_df.at[svid, 'prob']
        qual = feat_df.at[svid, 'qual']
        num = feat_df.at[svid, 'close_size']
        sv_info = row.tolist() + [num, prob, qual]
        outputs.append(sv_info)

    output_df = pd.DataFrame(outputs, columns=headers)
    return output_df

def assign_weighted_qual(row):
    weight = row['close_size']
    prob = row['prob']
    qual = math.tanh(weight * 0.2) * prob
    return qual


def load_sig_table(sigs_tsv, pos, svtype, flank):
    headers = ['chrom', 'start', 'end', 'type', 'len', 'qname', 'iden', 'sequence']
    sig_tbl = pd.read_csv(sigs_tsv, usecols=[0, 1, 2, 3, 4, 6, 8, 10], sep='\t', names=headers)
    if len(sig_tbl) == 0:
        return None
    sig_tbl['plat'] = ['HiFi' for _ in range(len(sig_tbl))]
    sig_tbl['len'] = sig_tbl['len'].abs()
    # sig_tbl = sig_tbl.loc[sig_tbl['type'] == svtype]
    sig_tbl = sig_tbl.loc[(abs(sig_tbl['start'] - pos) <= flank) & (sig_tbl['type'] == svtype)]

    ## Only have one or no signature reads that matches observed svtype
    if len(sig_tbl) < 2:
        return None
    # sig_tbl['dist'] = sig_tbl.apply(lambda row: int(row['start']) - pos, axis=1)
    sig_tbl['left_iden'] = sig_tbl.apply(lambda row: float(row['iden'].split(':')[0]), axis=1)
    sig_tbl['right_iden'] = sig_tbl.apply(lambda row: float(row['iden'].split(':')[1]), axis=1)
    # sig_tbl = sig_tbl.loc[(abs(sig_tbl['start'] - pos) <= flank) & (sig_tbl['type'] == svtype)]
    sig_tbl.drop(['iden'], axis=1, inplace=True)

    return sig_tbl


def fetch_depth(depth_tabix, chrom, start, end):
    reads = []
    for line in depth_tabix.fetch(chrom, start, end):
        entries = line.strip().split('\t')
        reads.append(int(entries[3]))

    if not reads:
        return 0, 0

    return round(np.mean(reads), 2), round(np.std(reads), 2)

def find_similar_cluster(obs_val, obs_pos, clusters):

    obs_sig_diff = []
    obs_size_diff = []

    for i, cluster in enumerate(clusters):
        # cluster_length = poa_size if poa_size != -1 else exp_size

        sig_len = [ele[4] for ele in cluster]
        # sig_pos_median = np.median([ele[1] for ele in cluster])
        mean_pseudo = np.mean(sig_len)
        obs_sig_diff.append([abs(mean_pseudo - obs_val), i])
        size_diff_ratio = abs(obs_val - mean_pseudo) / max(mean_pseudo, obs_val)
        obs_size_diff.append([size_diff_ratio, i])

        ## Changed v1.0
        # obs_sig_diff.append([abs(cluster_length - obs_val), i])
        # size_diff_ratio = abs(obs_val - cluster_length) / max(cluster_length, obs_val)
        # obs_size_diff.append([size_diff_ratio, i])


    ## Sort cluster by size and similarity, find the cluster with the most similar allele length
    min_size_diff, min_size_diff_idx = sorted(obs_size_diff, key=lambda x: x[0])[0]

    ## Penalty for finding the most similar signatures as the observed value
    close_cluster = clusters[min_size_diff_idx]

    # print(close_cluster[1:2])

    # close_cluster_sig_avg = np.mean([ele[0] for ele in close_cluster])
    close_cluster_sig_pos_mean = np.mean([ele[1] for ele in close_cluster])
    close_cluster_left_iden = np.median([ele[8] for ele in close_cluster])
    close_cluster_right_iden = np.median([ele[9] for ele in close_cluster])


    close_cluster_pos_penalty = round(abs(close_cluster_sig_pos_mean - obs_pos), 2)

    ## Only report cluster size where the size difference is smaller than 0.5.
    close_size = len(close_cluster) if min_size_diff <= 0.5 else 0
    # close_size = len(close_cluster)

    return close_cluster_pos_penalty, min_size_diff, close_cluster_left_iden, close_cluster_right_iden, close_size

def get_entropy(val_list):
    ent = 0
    sums = sum(val_list)
    for val in val_list:
        if val == 0:
            continue
        p = val / sums
        ent -= p * math.log2(p)
    return ent

def sig_tbl_hclust(svtype, sig_tbl, obs_pos, ref_fasta, max_pos_dist):
    cluster_max_distance = 0.5
    # sig_lists = sorted(sig_tbl.values.tolist(), key=lambda x: x[4])
    sig_lists = sig_tbl.values.tolist()
    distances = []
    for i in range(len(sig_lists) - 1):
        for j in range(i + 1, len(sig_lists)):
            distances.append(sig_dist(sig_lists[i][3], sig_lists[i], sig_lists[j]))
            ## Changed V1.0
            # distances.append(sig_dist1(sig_lists[i][3], ref_fasta, sig_lists[i], sig_lists[j]))

    Z = fastcluster.linkage(np.array(distances), method="average")
    cluster_indices = list(fcluster(Z, cluster_max_distance, criterion='distance'))
    new_clusters = [[] for i in range(max(cluster_indices))]
    for signature_index, cluster_index in enumerate(cluster_indices):
        ## append signature length, reference start position, left and right sequence identity.
        new_clusters[cluster_index - 1].append(sig_lists[signature_index])

    valid_clusters = []
    for i, cluster in enumerate(new_clusters):
        ## Only save clusters that within 1kbp
        clut_median_pos = np.median([ele[1] for ele in cluster])
        # clut_median_length = np.median([ele[4] for ele in cluster])
        if abs(clut_median_pos - obs_pos) < max_pos_dist and len(cluster) > 2:
            ## Changed V1.0
            # if svtype == 'INS':
            #     observed_size, realigned_size = realign_insertion_consensus(cluster, ref_fasta)
            #     # print(observed_size, realigned_size)
            # # print(len(cluster), observed_size, realigned_size)
            #     valid_clusters.append([cluster, observed_size, realigned_size])
            # else:
            #     valid_clusters.append([cluster, clut_median_length, clut_median_length])
            valid_clusters.append(cluster)

    ent = get_entropy(Counter(np.arange(len(valid_clusters))).values()) if len(valid_clusters) > 0 else -1
    return valid_clusters, ent


def sig_tbl_hclust_debug(svtype, sig_tbl, obs_pos, ref_fasta, max_pos_dist):
    import seaborn as sns
    import matplotlib.pylab as plt
    SVCOLOR = {'INS': '#3e549f', 'DEL': '#d93931', 'INV': '#5faf76', 'DUP': '#6b824c'}
    cluster_max_distance = 0.5
    sig_lists = sorted(sig_tbl.values.tolist(), key=lambda x: x[4])
    dist_array = np.zeros((len(sig_lists), len(sig_lists)))
    labels = []
    distances = []
    for i in range(len(sig_lists) - 1):
        for j in range(i + 1, len(sig_lists)):
            # distances.append(sig_dist(sig_lists[i][3], sig_lists[i], sig_lists[j]))
            ## Changed V1.0
            dist = sig_dist1(sig_lists[i][3], ref_fasta, sig_lists[i], sig_lists[j])
            distances.append(dist)
            dist_array[i][j] = dist

    row_colors = [SVCOLOR[ele] for ele in labels]
    sns.clustermap(dist_array, method='average', metric='euclidean',
                   # xticklabels=True,
                   # yticklabels=True,
                   row_cluster=False, cmap="YlGnBu", linewidth=.5, col_colors=row_colors)
    plt.show()

    Z = linkage(np.array(distances), method="average")
    cluster_indices = list(fcluster(Z, cluster_max_distance, criterion='distance'))
    new_clusters = [[] for i in range(max(cluster_indices))]
    for signature_index, cluster_index in enumerate(cluster_indices):
        ## append signature length, reference start position, left and right sequence identity.
        new_clusters[cluster_index - 1].append(sig_lists[signature_index])

    valid_clusters = []
    for i, cluster in enumerate(new_clusters):
        ## Only save clusters that within 1kbp
        clut_median_pos = np.median([ele[1] for ele in cluster])
        clut_median_length = np.median([ele[4] for ele in cluster])
        if abs(clut_median_pos - obs_pos) < max_pos_dist and len(cluster) > 2:
            ## Changed V1.0
            if svtype == 'INS':
                observed_size, realigned_size = realign_insertion_consensus(cluster, ref_fasta)
                # print(observed_size, realigned_size)
                # print(len(cluster), observed_size, realigned_size)
                valid_clusters.append([cluster, observed_size, realigned_size])
            else:
                valid_clusters.append([cluster, clut_median_length, clut_median_length])


def sig_dist(sigtype, sig1, sig2):

    span1, span2 = sig1[4], sig2[4]
    span_diff = abs(span1 - span2) / max(span1, span2)

    if sigtype == 'DEL':
        center1 = (sig1[1] + sig1[2]) // 2
        center2 = (sig2[1] + sig2[2]) // 2
        pos_diff = abs(center1 - center2) / 900
        return pos_diff + span_diff

    elif sigtype == 'INS':
        center1 = sig1[1]
        center2 = sig2[1]
        pos_diff = abs(center1 - center2) / 900
        return pos_diff + span_diff

## Added V1.0
def sig_dist1(sigtype, ref_fasta, sig1, sig2):
    if sigtype == 'DEL':
        span1, span2 = sig1[4], sig2[4]
        center1 = (sig1[1] + sig1[2]) // 2
        center2 = (sig2[1] + sig2[2]) // 2
        pos_diff = abs(center1 - center2) / 900
        span_diff = abs(span1 - span2) / max(span1, span2)

        return pos_diff + span_diff

    elif sigtype == 'INS':
        span1, span2 = sig1[4], sig2[4]
        center1 = sig1[1]
        center2 = sig2[1]
        pos_diff = abs(center1 - center2) / 900
        # span_diff = abs(span1 - span2) / max(span1, span2)
        edit_distance = compute_haplotype_edit_distance(sig1, sig2, ref_fasta)
        sequence_distance = edit_distance / max(span1, span2)


        return pos_diff + sequence_distance

## Added V1.0
def compute_haplotype_edit_distance(signature1, signature2, reference, window_padding = 100):
    window_start = min(signature1[1], signature2[1]) - window_padding
    window_end = max(signature1[1], signature2[1]) + window_padding

    #construct haplotype sequences for both signatures
    haplotype1 = reference.fetch(signature1[0], max(0, window_start), max(0, signature1[1])).upper()
    haplotype1 += signature1[6].upper()
    haplotype1 += reference.fetch(signature1[0], max(0, signature1[1]), max(0, window_end)).upper()

    haplotype2 = reference.fetch(signature2[0], max(0, window_start), max(0, signature2[1])).upper()
    haplotype2 += signature2[6].upper()
    haplotype2 += reference.fetch(signature2[0], max(0, signature2[1]), max(0, window_end)).upper()

    return align(haplotype1, haplotype2)["editDistance"]

## Added V1.0
def realign_insertion_consensus(ins_cluster, reference, window_padding = 100, maximum_haplotype_length = 10000, allowed_size_deviation = 2.0):
    #compute window containing all members of cluster
    member_start = [member[1] for member in ins_cluster]
    window_start = min(member_start) - window_padding
    window_end = max(member_start) + window_padding

    #construct haplotype sequences from all reads
    haplotypes = []
    for member in ins_cluster:
        haplotype_sequence = reference.fetch(member[0], max(0, window_start), max(0, member[1])).upper()
        haplotype_sequence += member[6].upper()
        haplotype_sequence += reference.fetch(member[0], max(0, member[1]), max(0, window_end)).upper()
        haplotypes.append(haplotype_sequence)
    # largest_haplotype_length = max([len(h) for h in haplotypes])
    mean_haplotype_length = np.mean([len(h) for h in haplotypes])

    consensus_reads, msa_reads = poa(haplotypes, algorithm=1, m=2, n=-4, g=-4, e=-2, q=-24, c=-1)

    # print(len(consensus_reads), mean_haplotype_length)
    #re-align consensus sequence to reference sequence in the window
    ref_sequence = reference.fetch(ins_cluster[0][0], max(0, window_start), max(0, window_end)).upper()

    consensus_reads_ref, msa_reads_ref = poa([consensus_reads, ref_sequence], algorithm=1, m=2, n=-4, g=-4, e=-2, q=-24, c=-1)

    # locate insertion relative to reference and check whether size is close to expected size
    expected_size = np.median([ele[4] for ele in ins_cluster])
    matches = []
    for match in re.finditer(r'-+', msa_reads_ref[1]):
        match_size = match.end() - match.start()
        size_ratio = max(match_size, expected_size) / min(match_size, expected_size)
        matches.append((match.start(), match_size, size_ratio))
    # good_matches = [m for m in matches if m[2] < allowed_size_deviation]

    realigned_size = -1

    if len(matches) == 1:
        realigned_size = matches[0][1]

    return expected_size, realigned_size

if __name__ == '__main__':

    import pysam
    svid = 'chr1-714324-DEL-50'
    ref = 'GRCh38'
    sample = 'NA12877'
    ref_genome = '/Volumes/eichler-vol28/eee_shared/assemblies/hg38/no_alt/hg38.no_alt.fa'
    chrom, svpos, svtype, svlen = svid.split('-')
    workdir = f'/Volumes/eichler-vol28/projects/medical_reference/nobackups/SVREF/platinum/{ref}/pedfilt_svs/tp_sv_sigs/{sample}/HiFi'
    # depth_bed = pysam.Tabixfile(f'{workdir}/svs.depth.bed.gz')

    sig_tbl = load_sig_table(f'{workdir}/{svid}.sigs.tsv', int(svpos), svtype, 10000)

    ref_fasta = pysam.FastaFile(ref_genome)

    # sig_tbl_hclust_debug(svtype, sig_tbl, int(svpos), ref_fasta, 5000)
    valid_clusters, ent = sig_tbl_hclust(svtype, sig_tbl, int(svpos), ref_fasta, 5000)
    find_similar_cluster(50, 714324, valid_clusters)


# def create_feat_matrix(workdir, svs_tbl, depth_bed, plat, ref):
#     no_sig = open(f'{workdir}/invalid_pred.txt', 'w')
#     feat_vec = []
#     failed = 0
#     for idx, row in svs_tbl.iterrows():
#         chrom, svpos, end = row['#CHROM'], int(row['POS']), int(row['END'])
#         svlen = abs(int(row['SVLEN']))
#         depth_mean, depth_std = fetch_depth(depth_bed, chrom, svpos - 50, end + 50)
#         svid = row['ID']
#         try:
#             sig_tbl = load_sig_table(f'{workdir}/{svid}.sigs.tsv', svpos, row['SVTYPE'], 10000)
#             if sig_tbl is None:
#                 print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\tNoSigReads\t{6}".format(
#                     row['#CHROM'], row['POS'], row['END'], svid, depth_mean, depth_std, plat), file=no_sig)
#                 # feat_vec.append([svid, plat, depth_mean, depth_std, -1, -1, -1, -1, -1, -1, row['SVTYPE'],
#                 #                  'NoSigReads', feat_code[ref], feat_code[plat]])
#                 failed += 1
#                 continue
#
#             if sig_tbl.empty:
#                 print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\tNoSigReads\t{6}".format(
#                     row['#CHROM'], row['POS'], row['END'], svid, depth_mean, depth_std, plat), file=no_sig)
#                 # feat_vec.append([svid, plat, depth_mean, depth_std, -1, -1, -1, -1, -1, -1, row['SVTYPE'],
#                 #                  'NoSigReads', feat_code[ref], feat_code[plat]])
#                 failed += 1
#                 continue
#
#             if len(sig_tbl) > 10000:
#                 logging.warning(f'{svid} in highly identical region, cannot predict')
#                 print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\tHighlyIdentical\t{6}".format(
#                     row['#CHROM'], row['POS'], row['END'], svid, depth_mean, depth_std, plat), file=no_sig)
#
#
#             clusters, cluster_ent = sig_tbl_hclust(sig_tbl, svpos, 5000)
#
#             if len(clusters) == 0:
#                 print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\tNoValidClusters\t{6}".format(
#                     row['#CHROM'], row['POS'], row['END'], svid, depth_mean, depth_std, plat), file=no_sig)
#                 # feat_vec.append([svid, plat, depth_mean, depth_std, -1, -1, -1, -1, -1, -1, row['SVTYPE'],
#                 #                  'NoValidClusters', feat_code[ref], feat_code[plat]])
#                 failed += 1
#                 continue
#
#             close_cluster_pos_penalty, close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden, close_cluster_size = find_similar_cluster(
#                 svlen, svpos, clusters)
#             feat_vec.append([svid, plat, depth_mean, depth_std, cluster_ent, close_cluster_pos_penalty,
#                              close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
#                              close_cluster_size, row['SVTYPE'], 'HasSigReads', feat_code[ref],
#                              feat_code[plat]])
#
#         except FileNotFoundError:
#             failed += 1
#             print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\tMissFile\t{6}".format(row['#CHROM'], row['POS'], row['END'], svid,
#                                                                        depth_mean, depth_std, plat), file=no_sig)
#
#     logging.info('******************** Create feature matrix ********************')
#     logging.info(f'No. of valid candidates: {len(feat_vec)}')
#     logging.info(f'Candidates without proper feature: {failed}')
#
#     feat_df = pd.DataFrame(feat_vec, columns=['svid', 'plat', 'depth_mean', 'depth_std', 'clusters_ent',
#                                               'close_pos_p', 'close_len_p', 'close_l_iden', 'close_r_iden',
#                                               'close_size', 'svtype', 'siginfo', 'ref_code', 'plat_code'])
#     return feat_df

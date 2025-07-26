#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jiadong Lin

@contact: jdlin@uw.edu

@time: 12/21/24
'''
import os
import multiprocessing
import argparse
import datetime
from time import strftime, localtime
import pysam
import logging
import pandas as pd

from src import Collect
from src import Predict
from src.version import __version__

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(help='modes', dest='sub')

    collect_parser = subparsers.add_parser('collect',
                                          help='Collect alignment and create feature matrix for SVs')
    collect_parser.add_argument('-o', dest='output', type=str, required=True,
                        help='Output directory')
    collect_parser.add_argument('-t', dest='threads', type=int, default=4,
                        help='Number of threads (default: %(default)s)')
    collect_parser.add_argument('-i', dest='bed_path', type=os.path.abspath, required=True,
                        help='Path to the SVs BED file')
    collect_parser.add_argument('-p', dest='info', type=str, required=True,
                        help='Platform and reference info seperated by comma (e.g. HiFi,GRCh38)')
    collect_parser.add_argument('-r', dest='reference', type=str, required=True,
                                help='Path to the reference genome')
    collect_parser.add_argument('-f', dest='flank', type=int, default=10000,
                        help='Size of flank regions for read analysis (default: %(default)s)')


    collect_parser.add_argument('-b', dest='bam_path', type=os.path.abspath, required=True,
                        help='Path to the BAM manifest file')

    collect_parser.add_argument('-d', dest='depth', type=os.path.abspath, required=True,
                                help='Read depth at reported SV regions')

    predict_parser = subparsers.add_parser('predict', help='Predict SV quality based on feature matrix')

    predict_parser.add_argument('-o', dest='output', type=str, required=True,
                        help='Output directory for predicted quality (default: pred_qual.txt)')
    predict_parser.add_argument('-s', dest='matrix', type=str, required=True,
                        help=f'Path to feature matrix file created in collect')

    predict_parser.add_argument('-i', dest='bed_path', type=os.path.abspath, required=True,
                        help='Path to the SVs BED file')
    predict_parser.add_argument('-p', dest='info', type=str, required=True,
                        help='Platform and reference info seperated by comma (e.g. HiFi,GRCh38)')

    # predict_parser.add_argument('-f', dest='flank', type=int, default=10000,
    #                     help='Size of flank regions for read analysis (default: %(default)s)')
    predict_parser.add_argument('-m', dest='model', type=str, required=True,
                        help='Path to the pretrained model')



    options = parser.parse_args()
    return options

def init_reading_process(depth_path, ref_path):
    global depth_bed
    global ref_fasta
    depth_bed = pysam.Tabixfile(depth_path)
    ref_fasta = pysam.FastaFile(ref_path)



def create_single_feat(workdir, plat, ref, svid, chrom, svpos, svlen, svtype):
    global depth_bed
    global ref_fasta
    # depth_bed = pysam.Tabixfile(f'{workdir}/svs.depth.bed.gz')
    feat_code = {'GRCh38': 1, 'CHM13': 0, 'HiFi': 1, 'ONT': 0}

    ## Get the read depth stats around the SV
    depth_mean, depth_std = -1, -1
    try:
        depth_mean, depth_std = Predict.fetch_depth(depth_bed, chrom, svpos - 50, svpos + svlen + 50)
    except ValueError:
        logging.warning(f'{svid} dose not have depth info')

    cluster_ent, close_cluster_pos_penalty, close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden, close_cluster_size = -1, -1, -1, -1, -1, -1
    sig_tbl = Predict.load_sig_table(f'{workdir}/{svid}.sigs.tsv', svpos, svtype, 10000)

    if sig_tbl is None:
        return [svid, plat, depth_mean, depth_std,  cluster_ent,  close_cluster_pos_penalty,
                             close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
                             close_cluster_size, svtype, 'NoSigReads', feat_code[ref], feat_code[plat]]

    if sig_tbl.empty:
        return [svid, plat, depth_mean, depth_std,  cluster_ent, close_cluster_pos_penalty,
                             close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
                             close_cluster_size, svtype, 'NoSigReads', feat_code[ref], feat_code[plat]]

    if len(sig_tbl) > 10000:
        logging.warning(f'{svid} in highly identical region, cannot predict')
        return [svid, plat, depth_mean, depth_std,  cluster_ent,close_cluster_pos_penalty,
                             close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
                             close_cluster_size, svtype, 'HighlyIdentical', feat_code[ref], feat_code[plat]]

    clusters, cluster_ent = Predict.sig_tbl_hclust(svtype, sig_tbl, svpos, ref_fasta,5000)

    if len(clusters) == 0:
        return [svid, plat, depth_mean, depth_std,  cluster_ent, close_cluster_pos_penalty,
                             close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
                             close_cluster_size, svtype, 'NoValidClusters', feat_code[ref], feat_code[plat]]

    close_cluster_pos_penalty, close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden, close_cluster_size = Predict.find_similar_cluster(
        svlen, svpos, clusters)

    # if close_cluster_pos_penalty == -1:
    #     return [svid, plat, depth_mean, depth_std, cluster_ent,
    #             close_cluster_pos_penalty, close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
    #             close_cluster_size, svtype, 'HasSigFailPOA', feat_code[ref], feat_code[plat]]

    return [svid, plat, depth_mean, depth_std, cluster_ent, close_cluster_pos_penalty,
                             close_cluster_len_penalty, close_cluster_left_iden, close_cluster_right_iden,
                             close_cluster_size, svtype, 'HasSigReads', feat_code[ref], feat_code[plat]]


if __name__ == '__main__':

    args = parse_arguments()

    workdir = args.output

    log_format = logging.Formatter("%(asctime)s [%(levelname)-7.7s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    start_time = datetime.datetime.now()


    svs_tbl = pd.read_csv(args.bed_path, sep='\t', header=[0])
    plat, ref = args.info.split(',')

    if args.sub == 'collect':
        flank = args.flank

        fileHandler = logging.FileHandler("{0}/BoostSV_Collect_{1}.log".format(workdir, strftime("%y%m%d_%H%M%S", localtime())),
                                          mode="w")
        fileHandler.setFormatter(log_format)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_format)
        root_logger.addHandler(fileHandler)

        logging.info('******************** Start collect features ********************')
        logging.info("INPUT SV: {0}".format(os.path.abspath(args.bed_path)))
        logging.info("INPUT BAM: {0}".format(os.path.abspath(args.bam_path)))
        logging.info("WORKDIR DIR: {0}".format(os.path.abspath(workdir)))

        try:
            f = open(args.bed_path)
        except FileNotFoundError:
            logging.error("SV BED file not found!!")
            exit()

        try:
            f = pysam.AlignmentFile(args.bam_path)
        except FileNotFoundError:
            logging.error("BAM file not found!!")
            exit()

        try:
            f = open(args.depth)
        except FileNotFoundError:
            logging.error("SV depth file not found!!")
            exit()


        collect_pool = multiprocessing.Pool(processes=args.threads)
        process_args = []
        recs = []
        ignored_svs = []
        for idx, row in svs_tbl.iterrows():
            chrom, start, end, svlen, svtype, svid = row['#CHROM'], row['POS'], row['END'], abs(int(row['SVLEN'])), row['SVTYPE'], row['ID']
            if chrom == 'chrY':
                ignored_svs.append(svid)
                continue
            if start < args.flank:
                logging.warning(f'{svid} Failed collection: start position smaller than flanking region size.')
                ignored_svs.append(svid)
                continue

            region_start = int(start) - flank
            region_end = int(start) + 1 + flank if svtype == 'INS' else int(start) + int(svlen) + flank

            # process_args.append((args, chrom, region_start, region_end, svid))
            recs.append([collect_pool.apply_async(Collect.collect_main, (args, chrom, region_start, region_end, svid))])

        collect_pool.close()
        collect_pool.join()

        logging.info('******************** Start create feature matrix ********************')

        predict_pool = multiprocessing.Pool(processes=args.threads, initializer=init_reading_process,
                                            initargs=(args.depth, args.reference, ))

        pred_args = []
        for idx, row in svs_tbl.iterrows():
            if row['ID'] in ignored_svs:
                continue
            pred_args.append((workdir, plat, ref, row['ID'], row['#CHROM'], row['POS'], abs(row['SVLEN']), row['SVTYPE']))
            # result = process_pool.starmap(Predict.create_single_feat, (workdir, plat, ref, row['ID'], depth_bed)).get()
            # print(result)
        results = predict_pool.starmap_async(create_single_feat, pred_args).get()
        # print(results)

        predict_pool.close()
        predict_pool.join()

        feat_df = pd.DataFrame(results, columns=['svid', 'plat', 'depth_mean', 'depth_std', 'clusters_ent',
                                                 'close_pos_p', 'close_len_p', 'close_l_iden', 'close_r_iden',
                                                 'close_size', 'svtype', 'siginfo', 'ref_code', 'plat_code'])

        feat_df.to_csv(f'{workdir}/sv_feats_v{__version__}.txt', sep='\t', header=True, index=False)


    elif args.sub == 'predict':

        feat_file = args.matrix

        fileHandler = logging.FileHandler("{0}/BoostSV_Predict_{1}.log".format(workdir, strftime("%y%m%d_%H%M%S", localtime())),mode="w")
        fileHandler.setFormatter(log_format)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_format)
        root_logger.addHandler(fileHandler)


        feat_df = pd.read_csv(feat_file, sep='\t', header=[0])

        ## Check duplicated SVs
        pred_df = feat_df.drop_duplicates(subset='svid', keep="last")

        valid_candi = pred_df.loc[pred_df['siginfo']=='HasSigReads']


        logging.info('Reading SV features:')
        logging.info(f'Total SVs: {len(feat_df)}')
        logging.info(f'Valid SV for prediction: {len(valid_candi)}')


        logging.info('******************** Start prediction ********************')

        pred_out = Predict.predict_main(svs_tbl, valid_candi, args.model)

        pred_out.to_csv(f'{workdir}/pred_qual_v{__version__}.txt', sep='\t', index=False, header=True)

        end_time = datetime.datetime.now()
        cost_time = (end_time - start_time).seconds / 60

        # print(f'Cost time: {round(cost_time, 2)} mins')
        logging.info(f"Complete collection, cost: {round(cost_time, 2)} mins")


#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jiadong Lin

@contact: jdlin@uw.edu

@time: 12/23/24
'''
import re
import pysam
from src.Signature import SignatureInsertion, SignatureDeletion

min_mapq = 10
segment_overlap_tolerance = 50
segment_gap_tolerance = 50
min_sv_size = 50
max_sv_size = 100000

def collect_main(options, chrom, start, end, svid):
    # global samfile
    samfile = pysam.AlignmentFile(options.bam_path)
    alignment_it = samfile.fetch(chrom, start, end)
    ref_fasta = pysam.FastaFile(options.reference)

    sv_signatures = []
    read_nr = 0
    read_dict = {}


    fout = open(f"{options.output}/{svid}.sigs.tsv", 'w')

    while True:
        try:
            current_alignment = next(alignment_it)

            if current_alignment.is_unmapped or current_alignment.is_secondary or current_alignment.mapping_quality < min_mapq:
                continue

            read_nr += 1

            if current_alignment.query_name not in read_dict:
                read_dict[current_alignment.query_name] = {'PM': [], 'SUPP': []}

            if current_alignment.is_supplementary:
                read_dict[current_alignment.query_name]['SUPP'].append(current_alignment)
            else:
                read_dict[current_alignment.query_name]['PM'].append(current_alignment)

            # read_dict[current_alignment.query_name]['SUPP'] = good_suppl_alns

        except StopIteration:
            break
        except KeyboardInterrupt:
            # logging.warning('Execution interrupted by user. Stop detection and continue with next step..')
            break

    for read, alignments in read_dict.items():
        good_suppl_alns = [aln for aln in alignments['SUPP'] if not aln.is_unmapped and aln.mapping_quality >= min_mapq]
        for align in alignments['PM'] + good_suppl_alns:
            cigar_indel = analyze_alignment_indel(align, samfile, ref_fasta, read)
            # sv_signatures.extend(cigar_indel)
            for sig in cigar_indel:
                if start < sig.start < end:
                    sig.set_iden()
                    sig.set_svid(svid)
                    print(sig.as_string(), file=fout)

        ## This read has primary alignment at this region
        if len(alignments['PM']) > 0:
            supp_insdel = analyze_read_segments(alignments['PM'][0], good_suppl_alns, ref_fasta, samfile)
            # sv_signatures.extend(sigs)
            for sig in supp_insdel:
                if start < sig.start < end:
                    sig.set_iden()
                    sig.set_svid(svid)
                    print(sig.as_string(), file=fout)
    fout.close()

def analyze_read_segments(primary, supplementaries, ref_fasta, bam):
    read_name = primary.query_name
    alignments = [primary] + supplementaries
    alignment_list = []
    for alignment in alignments:
        #correct query coordinates for reversely mapped reads
        if alignment.is_reverse:
            inferred_read_length = alignment.infer_read_length()
            if inferred_read_length is None:
                continue
            q_start = inferred_read_length - alignment.query_alignment_end
            q_end = inferred_read_length - alignment.query_alignment_start
        else:
            q_start = alignment.query_alignment_start
            q_end = alignment.query_alignment_end

        new_alignment_dict = {  'q_start': q_start,
                                'q_end': q_end,
                                'ref_id': alignment.reference_id,
                                'ref_start': alignment.reference_start,
                                'ref_end': alignment.reference_end,
                                'is_reverse': alignment.is_reverse,
                                'cigar': cigar_tuple(alignment.cigarstring),
                                'query_seq': alignment.query_sequence}

        alignment_list.append(new_alignment_dict)

    sorted_alignment_list = sorted(alignment_list, key=lambda aln: (aln['q_start'], aln['q_end']))
    #inferred_read_length = alignments[0].infer_read_length()

    sv_signatures = []
    #Translocation signatures from other SV classes are stored separately for --all_bnd option
    # translocation_signatures_all_bnds = []
    tandem_duplications = []

    for index in range(len(sorted_alignment_list) - 1):
        alignment_current = sorted_alignment_list[index]
        alignment_next = sorted_alignment_list[index + 1]

        distance_on_read = alignment_next['q_start'] - alignment_current['q_end']

        #Same chromosome
        if alignment_current['ref_id'] == alignment_next['ref_id']:
            ref_chr = bam.getrname(alignment_current['ref_id'])
            #Same orientation
            if alignment_current['is_reverse'] == alignment_next['is_reverse']:
                #Compute distance on reference depending on orientation
                if alignment_current['is_reverse']:
                    distance_on_reference = alignment_current['ref_start'] - alignment_next['ref_end']
                else:
                    distance_on_reference = alignment_next['ref_start'] - alignment_current['ref_end']
                #No overlap on read
                if distance_on_read >= -segment_overlap_tolerance:
                    #No overlap on reference
                    if distance_on_reference >= -segment_overlap_tolerance:
                        deviation = distance_on_read - distance_on_reference
                        #INS candidate
                        if deviation >= min_sv_size:
                            #No gap on reference
                            if distance_on_reference <= segment_gap_tolerance:
                                if not alignment_current['is_reverse']:
                                    try:
                                        insertion_seq = primary.query_sequence[alignment_current['q_end']:alignment_current['q_end']+deviation]
                                    except TypeError:
                                        insertion_seq = ""
                                    sv_signatures.append(SignatureInsertion(ref_chr, alignment_current['ref_end'], alignment_current['ref_end'] + deviation,
                                                                            deviation,"suppl-INS", [alignment_current['cigar'], alignment_next['cigar']],
                                                                            alignment_current['is_reverse'], read_name,insertion_seq))
                                else:
                                    try:
                                        insertion_seq = primary.query_sequence[primary.infer_read_length() - alignment_next['q_start']:primary.infer_read_length() - alignment_next['q_start'] + deviation]
                                    except TypeError:
                                        insertion_seq = ""
                                    sv_signatures.append(SignatureInsertion(ref_chr, alignment_current['ref_start'], alignment_current['ref_start'] + deviation,
                                                                            deviation,"suppl-INS", [alignment_next['cigar'], alignment_current['cigar']],
                                                                            alignment_current['is_reverse'], read_name, insertion_seq))
                        #DEL candidate
                        elif -max_sv_size <= deviation <= -min_sv_size:
                            #No gap on read
                            if distance_on_read <= segment_gap_tolerance:
                                deletion_seq = '' if ref_fasta == '' else ref_fasta.fetch(ref_chr, alignment_current['ref_end'], alignment_current['ref_end'] - deviation)
                                if not alignment_current['is_reverse']:
                                    sv_signatures.append(SignatureDeletion(ref_chr, alignment_current['ref_end'], alignment_current['ref_end'] - deviation,
                                                                           deviation,"suppl-DEL", [alignment_current['cigar'], alignment_next['cigar']],
                                                                           alignment_current['is_reverse'], read_name, deletion_seq))
                                else:
                                    sv_signatures.append(SignatureDeletion(ref_chr, alignment_next['ref_end'], alignment_next['ref_end'] - deviation,
                                                                           deviation,"suppl-DEL", [alignment_next['cigar'], alignment_current['cigar']], alignment_next['is_reverse'],read_name, deletion_seq))
                    #overlap on reference
                    else:
                        #Tandem Duplication, save as insertion
                        if distance_on_reference <= min_sv_size:
                            duplicate_seq = ref_fasta.fetch(ref_chr, alignment_next['ref_start'], alignment_current['ref_end'])
                            if not alignment_current['is_reverse']:
                                #Tandem Duplication
                                if alignment_next['ref_end'] > alignment_current['ref_start']:
                                    # tandem_duplications.append((ref_chr, alignment_next['ref_start'], alignment_current['ref_end'], True, True, alignment_current['cigar']))
                                    sv_signatures.append(SignatureInsertion(ref_chr, alignment_next['ref_start'], alignment_current['ref_end'], alignment_current['ref_end'] - alignment_next['ref_start'], "suppl-TD",
                                                                             [alignment_next['cigar'], alignment_current['cigar']], alignment_current['is_reverse'], read_name, duplicate_seq))
                            else:
                                #Tandem Duplication
                                if alignment_next['ref_start'] < alignment_current['ref_end']:
                                    # tandem_duplications.append((ref_chr, alignment_current['ref_start'], alignment_next['ref_end'], True, False, alignment_next['cigar']))
                                    sv_signatures.append(SignatureInsertion(ref_chr, alignment_current['ref_start'], alignment_next['ref_end'], alignment_next['ref_end'] - alignment_current['ref_start'], "suppl-TD",
                                                                            [alignment_current['cigar'], alignment_next['cigar']], alignment_current['is_reverse'] , read_name, duplicate_seq))
    return sv_signatures

def analyze_alignment_indel(alignment, bam, ref_fasta, query_name):
    sv_signatures = []
    #Translocation signatures from other SV classes are stored separately for --all_bnd option

    ref_chr = bam.getrname(alignment.reference_id)
    ref_start = alignment.reference_start
    indels = analyze_cigar_indel(alignment.cigartuples, min_sv_size)
    for pos_ref, pos_read, length, oper_idx, typ in indels:
        if typ == "DEL":
            deletion_seq = '' if ref_fasta == '' else ref_fasta.fetch(ref_chr, ref_start + pos_ref, ref_start + pos_ref + length)
            sv_signatures.append(SignatureDeletion(ref_chr, ref_start + pos_ref, ref_start + pos_ref + length,
                                                   length,f"cigar-{oper_idx}", cigar_tuple(alignment.cigarstring), alignment.is_reverse, query_name, deletion_seq))
        elif typ == "INS":
            try:
                insertion_seq = alignment.query_sequence[pos_read:pos_read+length]
            except TypeError:
                insertion_seq = ""
            sv_signatures.append(SignatureInsertion(ref_chr, ref_start + pos_ref, ref_start + pos_ref + length, length,
                                                    f"cigar-{oper_idx}", cigar_tuple(alignment.cigarstring), alignment.is_reverse, query_name, insertion_seq))
    return sv_signatures


def analyze_cigar_indel(tuples, min_length):
    """Parses CIGAR tuples (op, len) and returns Indels with a length > minLength"""
    pos_ref = 0
    pos_read = 0
    indels = []
    for i in range(len(tuples)):
        operation, length = tuples[i]
        if operation == 0:                     # alignment match
            pos_ref += length
            pos_read += length
        elif operation == 1:                   # insertion
            if length >= min_length:
                indels.append((pos_ref, pos_read, length, i, "INS"))
            pos_read += length
        elif operation == 2:                   # deletion
            if length >= min_length:
                indels.append((pos_ref, pos_read, length, i, "DEL"))
            pos_ref += length
        elif operation == 4:                   # soft clip
            pos_read += length
        elif operation == 7 or operation == 8:        # match or mismatch
            pos_ref += length
            pos_read += length
    return indels

def cigar_tuple(cigar):
    """
    Takes a cigar string as input and returns a cigar tuple
    """
    opVals = re.findall(r'(\d+)([\w=])', cigar)
    lengths = [int(opVals[i][0]) for i in range(0, len(opVals))]
    ops = [opVals[i][1] for i in range(0, len(opVals))]
    tuples = []
    for i in range(len(lengths)):
        tuples.append([ops[i], int(lengths[i])])
    return tuples
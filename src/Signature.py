#!/usr/bin/env python

# encoding: utf-8

'''

@author: Jiadong Lin

@contact: jdlin@uw.edu

@time: 12/21/24
'''


class Signature:
    """Signature class for basic signatures of structural variants. An signature is always detected from a single read.
    """

    def __init__(self, contig, start, end, length, signature, cigar, ori, read):
        self.contig = contig
        self.start = start
        self.end = end
        self.length = length
        self.signature = signature
        self.cigar = cigar
        self.read = read
        self.type = None
        self.ori = ori
        self.svid = '.'
        # if self.end < self.start:
        #     logging.warning("Signature with invalid coordinates (end < start): " + self.as_string())

    def set_svid(self, svid):
        self.svid = svid

    def get_source(self):
        return (self.contig, self.start, self.end)

    def get_key(self):
        contig, start, end = self.get_source()
        return (self.type, contig, end)

    def downstream_distance_to(self, signature2):
        """Return distance >= 0 between this signature's end and the start of signature2."""
        this_contig, this_start, this_end = self.get_source()
        other_contig, other_start, other_end = signature2.get_source()
        if self.type == signature2.type and this_contig == other_contig:
            return max(0, other_start - this_end)
        else:
            return float("inf")

    # def as_string(self, sep="\t"):
    #     contig, start, end = self.get_source()
    #     end = start + 1 if self.type == 'INS' else end
    #     ori = '-' if self.ori else '+'
    #     return sep.join(["{0}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}:{9}"]).format(
    #         contig, start, end, self.type, self.length, self.signature, self.read, ori, self.iden)


class SignatureDeletion(Signature):
    """SV Signature: a region (contig:start-end) has been deleted and is not present in sample"""

    def __init__(self, contig, start, end, length, signature, cigar, ori, read, sequence):
        self.contig = contig
        assert end >= start
        # 0-based start of the deletion (first deleted base)
        self.start = start
        # 0-based end of the deletion (one past the last deleted base)
        self.end = end
        self.length = length
        self.signature = signature
        self.read = read
        self.cigar = cigar
        self.type = "DEL"
        self.iden = [0, 0]
        self.ori = ori
        self.sequence = sequence

    def set_iden(self):
        if 'cigar' in self.signature:
            oper_idx = int(self.signature.split('-')[1])
            left_iden, right_iden = cigar_sig_identity(oper_idx, self.cigar, 500)
            self.iden = [left_iden, right_iden]

        else:
            iden, soft_idx = suppl_sig_identity(self.cigar[0], 500)
            self.iden[0] = iden
            iden, soft_idx = suppl_sig_identity(self.cigar[1], 500)
            self.iden[1] = iden


    def as_string(self, sep="\t"):
        contig, start, end = self.get_source()
        ori = '-' if self.ori else '+'
        return sep.join(["{0}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}:{9}", "{10}", "{11}"]).format(
            contig, start, end, self.type, self.length, self.signature, self.read, ori, self.iden[0], self.iden[1], self.svid, self.sequence)
class SignatureInsertion(Signature):
    """SV Signature: a region of length end-start has been inserted at contig:start"""

    def __init__(self, contig, start, end, length, signature, cigar, ori, read, sequence):
        self.contig = contig
        assert end >= start
        # 0-based start of the insertion (base after the insertion)
        self.start = start
        # 0-based start of the insertion (base after the insertion) + length of the insertion
        self.end = end
        self.signature = signature
        self.read = read
        self.sequence = sequence
        self.cigar = cigar
        self.length = length
        self.type = "INS"
        self.iden = [0, 0]
        self.ori = ori

    def get_key(self):
        contig, start, end = self.get_source()
        return (self.type, contig, start)


    def downstream_distance_to(self, signature2):
        """Return distance >= 0 between this signature's end and the start of signature2."""
        this_contig, this_start, this_end = self.get_source()
        other_contig, other_start, other_end = signature2.get_source()
        if self.type == signature2.type and this_contig == other_contig:
            return max(0, other_start - this_start)
        else:
            return float("inf")

    def set_iden(self):
        if 'cigar' in self.signature:
            oper_idx = int(self.signature.split('-')[1])
            left_iden, right_iden = cigar_sig_identity(oper_idx, self.cigar, 500)
            self.iden = [left_iden, right_iden]
        else:
            iden, soft_idx = suppl_sig_identity(self.cigar[0], 500)
            self.iden[0] = iden
            iden, soft_idx = suppl_sig_identity(self.cigar[1], 500)
            self.iden[1] = iden


    def as_string(self, sep="\t"):
        contig, start, end = self.get_source()
        ori = '-' if self.ori else '+'
        return sep.join(["{0}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}:{9}", "{10}", "{11}"]).format(
            contig, start, end, self.type, self.length, self.signature, self.read, ori, self.iden[0], self.iden[1], self.svid, self.sequence)


def suppl_sig_identity(cg_tuple, flank):
    operations = [i[0] for i in cg_tuple]
    clip_idx = operations.index('S') if 'S' in operations else operations.index('H')
    total_bases = 0
    mismatches = 0
    ## right ->
    if clip_idx == 0:
        for i in range(len(cg_tuple)):
            operation, length = cg_tuple[i]
            if total_bases <= flank:
                if operation == '=' or operation == 'M':
                    mismatches += length
                if operation in {'=', 'M', 'X', 'D', 'I'}:
                    total_bases += length
    ## <- left
    if clip_idx == len(cg_tuple) - 1:
        for i in range(len(cg_tuple)):
            operation, length = cg_tuple[len(cg_tuple) - i - 1]
            if total_bases <= flank:
                if operation == '=' or operation == 'M':
                    mismatches += length
                if operation in {'=', 'M', 'X', 'D', 'I'}:
                    total_bases += length

    return round(mismatches / total_bases * 100, 2), clip_idx

def cigar_sig_identity(oper_idx, cg_tuple, flank):

    left_total_bases, right_total_bases = 0, 0
    left_mismatches, right_mismatches = 0, 0

    for i in range(oper_idx+1, len(cg_tuple)):
        operation, length = cg_tuple[i]
        if right_total_bases <= flank:
            if operation == '=' or operation == 'M':
                right_mismatches += length
            if operation in {'=', 'M', 'X', 'D', 'I'}:
                right_total_bases += length
    for j in range(1, oper_idx):
        operation, length = cg_tuple[oper_idx - j]
        if left_total_bases <= flank:
            if operation == '=' or operation == 'M':
                left_mismatches += length
            if operation in {'=', 'M', 'X', 'D', 'I'}:
                left_total_bases += length

    left_iden = round(left_mismatches / left_total_bases * 100, 2)
    right_iden = round(right_mismatches / right_total_bases * 100, 2)
    return left_iden, right_iden



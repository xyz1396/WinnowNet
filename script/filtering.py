import getopt, sys, os
import numpy as np
import csv
import math
import re
from datetime import datetime, date, time
import pandas as pd
import sipros_post_module
import parseconfig
from sipros_peptides_assembling import *
import sys

## Glboal variables
pep_file_ext = '.pep.txt'
psm_file_ext = '.psm.txt'

pep_iden_str = '[Peptide_Identification]'
fasta_database_str = 'FASTA_Database'
pro_iden_str = '[Protein_Identification]'
testing_decoy_prefix_str = 'Testing_Decoy_Prefix'
training_decoy_prefix_str = 'Training_Decoy_Prefix'
reserved_decoy_prefix_str = 'Reserved_Decoy_Prefix'
min_peptide_per_protein_str = 'Min_Peptide_Per_Protein'
min_unique_peptide_per_protein_str = 'Min_Unique_Peptide_Per_Protein'
remove_decoy_identification_str = 'Remove_Decoy_Identification'
search_type_str = 'Search_Type'

sipros4_psmout_column_length = 20
sipros4_input = None

# defaul value
decoy_prefix = 'Rev_'
min_peptide_per_protein = 2
min_unique_peptide_per_protein = 1
remove_decoy_identification = 'No'

class PSM:
    def __init__(self, filename, file, scan, ParentCharge, rank, MeasuredParentMass,CalculatedParentMass,Massdiff, rescore, PTM_score, IdentifiedPeptide,PSM_Label,
                 Proteins,Proteinname,ProteinCount):
        self.filename=filename
        self.file = file
        self.scan = scan
        self.ParentCharge = ParentCharge
        self.rank = rank
        self.MeasuredParentMass=MeasuredParentMass
        self.CalculatedParentMass=CalculatedParentMass
        self.Massdiff = Massdiff
        self.MassErrorPPM='NA'
        self.ScanType='HCD'
        self.SearchName='Deep learning'
        self.ScoringFunction='softmax'
        self.rescore = rescore
        self.DeltaZ='NA'
        self.DeltaP='NA'
        self.PTM_score = PTM_score
        self.IdentifiedPeptide = IdentifiedPeptide
        self.OriginalPeptide='NA'
        self.PSM_Label = PSM_Label
        self.Proteins = Proteins
        self.Proteinname=Proteinname
        self.ProteinCount=ProteinCount


class Peptide:
    def __init__(self):
        self.IdentifiedPeptide = ''
        self.ParentCharge = ''
        self.OriginalPeptide = ''
        self.ProteinNames = []
        self.ProteinCount = 0
        self.SpectralCount = 0
        self.BestScore = 0.0
        self.PSMs = []

    def add(self, oPsm):
        self.SpectralCount += 1
        if self.BestScore < oPsm.rescore:
            self.BestScore = oPsm.rescore
        self.PSMs.append('{0}_{1}_{2}_{3}'.format(oPsm.file, oPsm.scan, oPsm.ParentCharge, oPsm.rank))
        self.ScanType = 'HCD'
        self.SearchName = 'Deep learning'
        if oPsm.PSM_Label==True:
            self.TargetMatch='T'
        else:
            self.TargetMatch='F'

    def set(self, oPsm):
        self.IdentifiedPeptide = oPsm.IdentifiedPeptide
        self.ParentCharge = oPsm.ParentCharge
        self.OriginalPeptide = oPsm.OriginalPeptide
        self.ProteinNames = oPsm.Proteinname
        self.ProteinCount = oPsm.ProteinCount
        self.SpectralCount = 1
        self.BestScore = oPsm.rescore
        self.PSMs.append('{0}_{1}_{2}_{3}'.format(oPsm.file, oPsm.scan, oPsm.ParentCharge, oPsm.rank))
        self.ScanType = 'HCD'
        self.SearchName = 'Deep learning'
        if oPsm.PSM_Label==True:
            self.TargetMatch = 'T'
        else:
            self.TargetMatch = 'F'


# # Division error handling
divide = sipros_post_module.divide
FDR_parameter = 1.0


# # FDR calculator
def FDR_calculator(FP, TP):
    FDR_numerator = float(FP) * float(FDR_parameter)
    FDR_denominator = float(TP)
    FDR_accept = True

    if FDR_denominator == 0:
        FDR_value = 1.0
        FDR_accept = False
    else:
        FDR_value = divide(FDR_numerator, FDR_denominator)
        FDR_accept = True

    return (FDR_accept, float(FDR_value))


def show_Fdr(psm_list, fdr_float, charge_left_given=-1, charge_right_given=-1):
    # list_sorted = sorted(psm_list, key=lambda x: (x.fPredictProbability, 1 - x.fRankProduct) , reverse=True)
    list_sorted = sorted(psm_list, key=lambda psm: (psm.rescore, psm.Massdiff, psm.PTM_score), reverse=True)
    decoy = 0
    target = 0
    best_nums = [0, 0]

    psm_filtered_list = []
    cutoff_probability = 1000.0
    # without considering training label
    for oPsm in list_sorted:
        '''
        if charge_left_given != -1 and (
                oPsm.ParentCharge < charge_left_given or oPsm.ParentCharge > charge_right_given):
            continue
        '''
        if oPsm.PSM_Label:
            target += 1
        elif not oPsm.PSM_Label:
            decoy += 1
        else:
            sys.stderr.write('error 768.\n')
        (FDR_accept, FDR_value) = FDR_calculator(decoy, target)
        if (FDR_accept is True) and (FDR_value <= fdr_float):
            if (best_nums[0] + best_nums[1]) < (decoy + target):
                best_nums = [decoy, target]
                cutoff_probability = oPsm.rescore

    for oPsm in list_sorted:
        if charge_left_given != -1 and (
                oPsm.ParentCharge < charge_left_given or oPsm.ParentCharge > charge_right_given):
            continue
        if oPsm.rescore >= cutoff_probability:
            psm_filtered_list.append(oPsm)

    return psm_filtered_list


## peptide level filtering
def show_Fdr_Pep(psm_list, fdr_float, charge_left_given=-1, charge_right_given=-1):
    list_sorted = sorted(psm_list, key=lambda psm: (psm.rescore, psm.Massdiff, psm.PTM_score), reverse=True)

    peptide_set = set()
    decoy = 0
    target = 0
    best_nums = [0, 0]

    psm_filtered_list = []
    cutoff_probability = 1000.0
    # without considering training label
    for oPsm in list_sorted:
        '''
        if charge_left_given != -1 and (
                oPsm.ParentCharge < charge_left_given or oPsm.ParentCharge > charge_right_given):
            continue
        '''
        pep_str = oPsm.IdentifiedPeptide + '_' + str(oPsm.ParentCharge)
        if pep_str not in peptide_set:
            if oPsm.PSM_Label:
                target += 1
                peptide_set.add(pep_str)
            elif not oPsm.PSM_Label:
                decoy += 1
                peptide_set.add(pep_str)
            else:
                sys.stderr.write('error 768.\n')

        (FDR_accept, FDR_value) = FDR_calculator(decoy, target)
        if (FDR_accept is True) and (FDR_value <= fdr_float):
            if (best_nums[0] + best_nums[1]) < (decoy + target):
                best_nums = [decoy, target]
                cutoff_probability = oPsm.rescore

    peptide = dict()
    for oPsm in list_sorted:
        '''
        if charge_left_given != -1 and (
                oPsm.ParentCharge < charge_left_given or oPsm.ParentCharge > charge_right_given):
            continue
        '''
        pep_str=oPsm.IdentifiedPeptide+'_'+str(oPsm.ParentCharge)
        #pep_str = oPsm.IdentifiedPeptide
        if oPsm.rescore >= cutoff_probability:
            if pep_str in peptide:
                peptide[pep_str].add(oPsm)
            else:
                oPeptide=Peptide()
                oPeptide.set(oPsm)
                peptide[pep_str]=oPeptide

    # return set(psm_filtered_list)
    return peptide


## remove redundant psm, only one unique spectrum kept
def re_rank(psm_list, consider_charge_bool=False):
    psm_new_list = []
    psm_dict = {}
    if consider_charge_bool:
        for oPsm in psm_list:
            sId = '{0}_{1}_{2}'.format(str(oPsm.file), str(oPsm.scan), str(oPsm.ParentCharge))
            if sId in psm_dict:
                if oPsm.rescore > psm_dict[sId].rescore:
                    psm_dict[sId] = oPsm
                elif oPsm.rescore == psm_dict[sId].rescore:
                    if abs(oPsm.Massdiff) < abs(psm_dict[sId].Massdiff):
                        psm_dict[sId] = oPsm
                    elif abs(oPsm.Massdiff) == abs(psm_dict[sId].Massdiff):
                        # calculate PTM scores
                        if oPsm.PTM_score < psm_dict[sId].PTM_score:
                            psm_dict[sId] = oPsm
                        elif oPsm.PTM_score == psm_dict[sId].PTM_score:
                            if oPsm.IdentifiedPeptide.upper() < psm_dict[sId].IdentifiedPeptide.upper():
                                psm_dict[sId] = oPsm
                            elif oPsm.IdentifiedPeptide.upper() == psm_dict[sId].IdentifiedPeptide.upper():
                                psm_dict[sId].add_protein(oPsm.protein_list)

            else:
                psm_dict[sId] = oPsm
    else:
        for oPsm in psm_list:
            sId = '{0}_{1}'.format(str(oPsm.file), str(oPsm.scan))
            if sId in psm_dict:
                if oPsm.rescore > psm_dict[sId].rescore:
                    psm_dict[sId] = oPsm
                elif oPsm.rescore == psm_dict[sId].rescore:
                    if abs(oPsm.Massdiff) < abs(psm_dict[sId].Massdiff):
                        psm_dict[sId] = oPsm
                    elif abs(oPsm.Massdiff) == abs(psm_dict[sId].Massdiff):
                        # calculate PTM scores
                        if oPsm.PTM_score < psm_dict[sId].PTM_score:
                            psm_dict[sId] = oPsm
                        elif oPsm.PTM_score == psm_dict[sId].PTM_score:
                            if oPsm.IdentifiedPeptide.upper() < psm_dict[sId].IdentifiedPeptide.upper():
                                psm_dict[sId] = oPsm


            else:
                psm_dict[sId] = oPsm

    for _key, value in psm_dict.items():
        psm_new_list.append(value)

    return psm_new_list


def cometToDict(filename):
    cometdict = dict()
    fp = open(filename)
    for line_id, line in enumerate(fp):
        if line_id == 0:
            continue
        s = line.strip().split('\t')
        length = len(s)
        idx = s[0]
        string = idx.split('_')
            
        filename='{0}_{1}_{2}_{3}_{4}_{5}.{6}'.format(string[0], string[1], string[2], string[3],string[4],string[5],'ms2')
        file_id = str(int(string[-4]))
        scan = str(int(string[-3]))
        charge = str(int(string[-2]))
        rank = str(int(string[-1]))
        '''
        filename='{0}_{1}.{2}'.format(string[0], string[1],'ms2')
        file_id = str(int(string[1]))
        scan = str(int(string[2]))
        charge = str(int(string[3]))
        rank = str(int(string[4]))
            
        filename = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.{9}'.format(string[0], string[1], string[2], string[3], string[4],string[5],string[6], string[7], string[8], string[9],'ms2')
        if int(string[8].replace('Run','')) == 1:
            file_id=str(int(string[9]))
        else:
            file_id=str(int(string[9])+11)
        scan = str(int(string[10]))
        charge = str(int(string[11]))
        rank = str(int(string[12]))
        '''

        idx = '{0}_{1}_{2}_{3}'.format(file_id, scan, charge, rank)
        MeasuredParentMass=float(s[3])
        CalculatedParentMass=float(s[4])
        MassDiff = abs(float(s[4]) - float(s[3]))
        IdentifyPeptide = s[26].replace('[15.9949]', '~').split('.')[1]
        PTM_score = IdentifyPeptide.count('~')
        Xcorr = float(s[9])
        Proteinname=s[27]
        ProteinCount=0

        Proteins = []
        for pidx in range(27, length):
            if s[pidx] != '':
                Proteins.append(s[pidx])
                if pidx==27:
                    continue
                Proteinname+=','+s[pidx]
        PSM_Label = False
        for protein in Proteins:
            if 'Rev' not in protein:
                ProteinCount+=1
                PSM_Label = True
                break

        cometdict[idx] = [filename,file_id, scan, charge, rank, MeasuredParentMass,CalculatedParentMass, MassDiff, Xcorr, PTM_score, IdentifyPeptide, PSM_Label,
                              Proteins,Proteinname,ProteinCount]

    fp.close()

    return cometdict


def readData(filename,filename2):
    cometdic = cometToDict(filename2)
    PSMs = []
    f = open(filename)
    for line_id,line in enumerate(f):
        line = line.strip().split(':')
        if line[0] in cometdic.keys():
            scan = cometdic[line[0]]
            PSMs.append(PSM(scan[0], int(scan[1]), int(scan[2]), int(scan[3]), int(scan[4]), float(scan[5]),float(scan[6]),float(scan[7]),float(line[1]),int(scan[9]), scan[10], scan[11], scan[12],scan[13],scan[14]))

    f.close()

    return PSMs

'''
def readqrankerData():
    PSMs = []
    for i in range(1,23):
        #f = open('./qranker/marine3/marine3_' + str(i) + '.csv')
        #f = open('crux-output/soil3_p'+str(i)+'_qranker.txt')
        f = open('./qranker/ecoli/ecoli_' + str(i) + '.csv')
        for line_id, line in enumerate(f):
            if line_id==0:
                continue
            s = line.strip().split('\t')
            filename=s[20]
            #file_id = str(int(filename.split('_')[0].replace('soil','')))
            #file_id = str(int(filename.replace('.sqt', '').replace('marine','')))
            file_id=i
            scan = str(int(s[0]))
            charge = str(int(s[1]))
            rank = str(int(s[12]))
            idx = '{0}_{1}_{2}_{3}'.format(file_id, scan, charge, rank)
            MeasuredParentMass=float(s[6])
            CalculatedParentMass=float(s[7])
            MassDiff = abs(float(s[6]) - float(s[7]))
            IdentifyPeptide = s[16].replace('[15.9949]', '~')
            PTM_score = IdentifyPeptide.count('~')
            Xcorr = float(s[2])
            Proteinname=s[18]
            ProteinCount=0

            Proteins = Proteinname.strip().split(',')
            PSM_Label = False
            for protein in Proteins:
                if 'Rev' not in protein:
                    ProteinCount+=1
                    PSM_Label = True
                    break

            PSMs.append(PSM(filename,file_id, scan, charge, rank, MeasuredParentMass,CalculatedParentMass, MassDiff, -Xcorr, PTM_score, IdentifyPeptide, PSM_Label,
                              Proteins,Proteinname,ProteinCount))



    return PSMs

'''
def readqrankerData():
    PSMs = []
    #f = open('./qranker/marine3/marine3_' + str(i) + '.csv')
    f = open('qranker/marine3_qranker.txt')
    for line_id, line in enumerate(f):
        if line_id==0:
            continue
        s = line.strip().split('\t')
        filename=s[20]
        #file_id = str(int(filename.split('_')[5].replace('.sqt','')))
        file_id = str(int(filename.replace('.sqt', '').replace('marine','')))
        scan = str(int(s[0]))
        charge = str(int(s[1]))
        rank = str(int(s[12]))
        idx = '{0}_{1}_{2}_{3}'.format(file_id, scan, charge, rank)
        MeasuredParentMass=float(s[6])
        CalculatedParentMass=float(s[7])
        MassDiff = abs(float(s[6]) - float(s[7]))
        IdentifyPeptide = s[16].replace('[15.9949]', '~')
        PTM_score = IdentifyPeptide.count('~')
        Xcorr = float(s[2])
        Proteinname=s[18]
        ProteinCount=0

        Proteins = Proteinname.strip().split(',')
        PSM_Label = False
        for protein in Proteins:
            if 'Rev' not in protein:
                ProteinCount+=1
                PSM_Label = True
                break

        PSMs.append(PSM(filename,file_id, scan, charge, rank, MeasuredParentMass,CalculatedParentMass, MassDiff, -Xcorr, PTM_score, IdentifyPeptide, PSM_Label,
                              Proteins,Proteinname,ProteinCount))



    return PSMs

'''


def readqrankerData():
    PSMs = []
    #f = open('./qranker/marine3/marine3_' + str(i) + '.csv')
    f = open('crux-output/marine2_1_qranker.txt')
    for line_id, line in enumerate(f):
        if line_id==0:
            continue
        s = line.strip().split('\t')
        filename=s[20]
        #file_id = str(int(filename.split('_')[5].replace('.sqt','')))
        file_id = str(int(filename.replace('.sqt', '').replace('marine','')))
        
        scan = str(int(s[0]))
        charge = str(int(s[1]))
        rank = str(int(s[12]))
        idx = '{0}_{1}_{2}_{3}'.format(file_id, scan, charge, rank)
        MeasuredParentMass=float(s[6])
        CalculatedParentMass=float(s[7])
        MassDiff = abs(float(s[6]) - float(s[7]))
        IdentifyPeptide = s[16].replace('[15.9949]', '~')
        PTM_score = IdentifyPeptide.count('~')
        Xcorr = float(s[2])
        Proteinname=s[18]
        ProteinCount=0

        Proteins = Proteinname.strip().split(',')
        PSM_Label = False
        for protein in Proteins:
            if 'Rev' not in protein:
                ProteinCount+=1
                PSM_Label = True
                break

        PSMs.append(PSM(filename,file_id, scan, charge, rank, MeasuredParentMass,CalculatedParentMass, MassDiff, -Xcorr, PTM_score, IdentifyPeptide, PSM_Label,
                              Proteins,Proteinname,ProteinCount))
    f = open('crux-output/marine2_2_qranker.txt')
    for line_id, line in enumerate(f):
        if line_id==0:
            continue
        s = line.strip().split('\t')
        filename=s[20]
        file_id = str(int(filename.replace('.sqt', '').replace('marine','')))
        #file_id = str(int(filename.replace('.sqt', '').replace('marine','')))
        scan = str(int(s[0]))
        charge = str(int(s[1]))
        rank = str(int(s[12]))
        idx = '{0}_{1}_{2}_{3}'.format(file_id, scan, charge, rank)
        MeasuredParentMass=float(s[6])
        CalculatedParentMass=float(s[7])
        MassDiff = abs(float(s[6]) - float(s[7]))
        IdentifyPeptide = s[16].replace('[15.9949]', '~')
        PTM_score = IdentifyPeptide.count('~')
        Xcorr = float(s[2])
        Proteinname=s[18]
        ProteinCount=0

        Proteins = Proteinname.strip().split(',')
        PSM_Label = False
        for protein in Proteins:
            if 'Rev' not in protein:
                ProteinCount+=1
                PSM_Label = True
                break

        PSMs.append(PSM(filename,file_id, scan, charge, rank, MeasuredParentMass,CalculatedParentMass, MassDiff, -Xcorr, PTM_score, IdentifyPeptide, PSM_Label,
                              Proteins,Proteinname,ProteinCount))


    return PSMs
'''


def readPercolatorData():
    PSMs = []
    f = open('percolator/OSU1.csv')
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        s = line.strip().split('\t')
        length = len(s)
        idx = s[0].split('_')
        '''
        string = idx
        filename = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.{9}'.format(string[0], string[1], string[2], string[3], string[4],string[5],string[6], string[7], string[8], string[9],'ms2')
        
        if int(string[8].replace('Run','')) == 1:
            file_id=str(int(string[9]))
        else:
            file_id=str(int(string[9])+11)
        scan = str(int(string[10]))
        charge = str(int(string[11]))
        rank = str(int(string[12]))
        '''
        '''
        string = idx
        filename = '{0}_{1}_{2}_{3}_{4}_{5}.{6}'.format(string[0], string[1], string[2], string[3], string[4],string[5], 'ms2')
        file_id = str(int(string[5]))
        scan = str(int(string[6]))
        charge = str(int(string[7]))
        rank = str(int(string[8]))
        
        '''
        string = idx
        filename = '{0}_{1}.{2}'.format(string[0], string[1], 'ms2')
        file_id = str(int(string[1]))
        i=int(file_id)

        scan = str(int(string[2]))
        charge = str(int(string[3]))
        rank = str(int(string[4]))
        
        idx = '{0}_{1}_{2}_{3}'.format(file_id, scan, charge, rank)
        IdentifyPeptide = s[4].replace('[15.9949]', '~').split('.')[1]
        PTM_score = IdentifyPeptide.count('~')

        Proteinname = s[5]
        ProteinCount = 0

        Proteins = []
        for pidx in range(5, length):
            if s[pidx] != '':
                Proteins.append(s[pidx])
                if pidx == 5:
                    continue
                Proteinname += ',' + s[pidx]
        PSM_Label = False
        for protein in Proteins:
            if 'Rev' not in protein:
                ProteinCount += 1
                PSM_Label = True
                break


        PSMs.append(
            PSM(filename, int(file_id), int(scan), int(charge),int(rank), 0.0, 0.0,0.0,float(s[1]), PTM_score, IdentifyPeptide, PSM_Label, Proteins,Proteinname,ProteinCount))
    return PSMs

def readWinnowNetData(singlefile,rescore):
    PSMs = []
    rescores = []
    with open(rescore) as f:
        for line in f:
            rescores.append(float(line.strip()))
    f = open(singlefile)
    for line_id, line in enumerate(f):
        s = line.strip().split('\t')
        length = len(s)
        string = s[0].split('_')
        filename = '_'.join(string[:-3])
        file_id = filename
        scan = str(int(string[-3]))
        charge = str(int(string[-2]))
        rank = str(int(string[-1]))
        if str(int(string[-1]))!='1':
            continue
        IdentifyPeptide = s[1]
        PTM_score = IdentifyPeptide.count('~')

        Proteinname = s[2]
        ProteinCount = 0

        Proteins = []
        for pidx in range(2, length):
            if s[pidx]!='':
                Proteins.append(s[pidx])
                if pidx == 2:
                    continue
                Proteinname += ',' + s[pidx]
        PSM_Label = False
        for protein in Proteins:
            if decoy_prefix not in protein:
                ProteinCount += 1
                PSM_Label = True
                break
        PSMs.append(PSM(filename, file_id, int(scan), int(charge),int(rank), 0.0, 0.0,0.0,float(rescores[line_id]), PTM_score, IdentifyPeptide, PSM_Label, Proteins,Proteinname,ProteinCount))
    return PSMs



def readCometData():
    cometdic = cometToDict()
    PSMs = []
    for i in range(1, 12):
        
        if i < 10:
            fp = open('comet/marine3/OSU_D7_FASP_Elite_03172014_0'+str(i)+'.txt')
        else:
            fp = open('comet/marine3/OSU_D7_FASP_Elite_03172014_'+str(i)+'.txt')
        
        #fp = open('comet/soil3/soil3_' + str(i) + '.txt')
        
        filename = fp.name
        for line_id, line in enumerate(fp):
            if line_id == 0:
                continue
            if line_id == 1:
                continue
            s = line.strip().split('\t')
            file_id=i
            scan=int(s[0])
            rank=int(s[1])
            charge=int(s[2])
            length = len(s)
            MeasuredParentMass=float(s[3])
            CalculatedParentMass=float(s[4])
            Massdiff = abs(float(s[3]) - float(s[4]))
            rescore = float(s[5])
            IdentifyPeptide = s[12].replace('[15.9949]', '~').split('.')[1]
            PTM_score = IdentifyPeptide.count('~')
            Proteinname=s[15]
            pro = s[15].strip().split(',')
            ProteinCount = 0
            Proteins = []
            for pidx in range(0, len(pro)):
                if pro[pidx] != '':
                    Proteins.append(pro[pidx])
            PSM_Label = False
            for protein in Proteins:
                if 'Rev' not in protein:
                    ProteinCount += 1
                    PSM_Label = True
                    break
            PSMs.append(
                PSM(filename, file_id, scan, charge, rank, MeasuredParentMass, CalculatedParentMass, Massdiff, -rescore,
                    PTM_score, IdentifyPeptide, PSM_Label,
                    Proteins, Proteinname, ProteinCount))
    return PSMs


def read_iprophet(input_file, mix_version=False):
    PSMs=[]
    C_pattern = re.compile(r'C\[160\]')
    M_pattern = re.compile(r'M\[147\]')
    clean_pattern = re.compile('[">/]')
    scan_id = 0
    charge_id = ''
    original_pep = ''
    identified_pep = ''
    protein_l = []
    iProbability = 0.0
    ntt = 0
    nmc = 0
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("<spectrum_query "):
                count=0
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('spectrum='):
                        split_l_2 = one.split('=')
                        filename=split_l_2[-1].split('.')[0].replace('"','')+'.ms2'
                        prefix=split_l_2[-1].split('.')[0].split('_')
                        if int(prefix[-2].replace('Run',''))==1:
                            file_id=int(prefix[-1])
                        else:
                            file_id=int(prefix[-1])+11
                    if one.startswith('start_scan='):
                        split_l_2 = one.split('=')
                        scan_id = int(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith('precursor_neutral_mass='):
                        split_l_2 = one.split('=')
                        MeasuredParentMass= float(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith('assumed_charge='):
                        split_l_2 = one.split('=')
                        charge_id = clean_pattern.sub('', split_l_2[-1])

                protein_l = []
                ntt = 2
                nmc = 0
            if line.startswith("<parameter name=\"ntt\""):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('value='):
                        split_l_2 = one.split('=')
                        ntt = int(clean_pattern.sub('', split_l_2[-1]))
            if line.startswith("<parameter name=\"nmc\""):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('value='):
                        split_l_2 = one.split('=')
                        nmc = int(clean_pattern.sub('', split_l_2[-1]))

            if line.startswith("<search_hit"):
                count+=1
                if count>1:
                    continue
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('peptide='):
                        split_l_2 = one.split('=')
                        original_pep = clean_pattern.sub('', split_l_2[-1])
                        identified_pep = original_pep
                        PTM_score=identified_pep.count('~')
                    if one.startswith("protein="):
                        split_l_2 = one.split('=')
                        protein_l.append(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith("calc_neutral_pep_mass="):
                        split_l_2 = one.split('=')
                        CalculatedParentMass=float(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith("massdiff="):
                        split_l_2 = one.split('=')
                        MassDiff=float(clean_pattern.sub('', split_l_2[-1]))

            if line.startswith("<modification_info modified_peptide"):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('modified_peptide='):
                        split_l_2 = one.split('=')
                        identified_pep = C_pattern.sub('C', (clean_pattern.sub('', split_l_2[-1])))
                        identified_pep = M_pattern.sub('M~', (clean_pattern.sub('', split_l_2[-1])))
            if line.startswith("<alternative_protein"):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('protein='):
                        split_l_2 = one.split('=')
                        tmp_str = clean_pattern.sub('', split_l_2[-1])
                        if tmp_str not in protein_l:
                            protein_l.append(tmp_str)
            if line.startswith("<interprophet_result "):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('probability='):
                        split_l_2 = one.split('=')
                        iProbability = float(clean_pattern.sub('', split_l_2[-1]))
            if line.startswith("</spectrum_query>"):
                PSM_Label = False
                for p in protein_l:
                    if 'Rev' not in p:
                        PSM_Label = True
                        break
                PSMs.append(PSM(filename,file_id,scan_id,charge_id,'NA',MeasuredParentMass, CalculatedParentMass, MassDiff,iProbability,PTM_score,identified_pep,PSM_Label,protein_l,','.join(protein_l),len(protein_l)))
                protein_l = []


    return PSMs

def read_prophet(input_file, mix_version=False):
    PSMs=[]
    C_pattern = re.compile(r'C\[160\]')
    M_pattern = re.compile(r'M\[147\]')
    clean_pattern = re.compile('[">/]')
    scan_id = 0
    charge_id = ''
    original_pep = ''
    identified_pep = ''
    protein_l = []
    iProbability = 0.0
    ntt = 0
    nmc = 0
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("<spectrum_query "):
                count=0
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('spectrum='):
                        split_l_2 = one.split('=')
                        filename=split_l_2[-1].split('.')[0].replace('"','')+'.ms2'
                        prefix=split_l_2[-1].split('.')[0].split('_')
                        if int(prefix[-2].replace('Run',''))==1:
                            file_id=int(prefix[-1])
                        else:
                            file_id=int(prefix[-1])+11
                    if one.startswith('start_scan='):
                        split_l_2 = one.split('=')
                        scan_id = int(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith('precursor_neutral_mass='):
                        split_l_2 = one.split('=')
                        MeasuredParentMass= float(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith('assumed_charge='):
                        split_l_2 = one.split('=')
                        charge_id = clean_pattern.sub('', split_l_2[-1])

                protein_l = []
                ntt = 2
                nmc = 0
            if line.startswith("<parameter name=\"ntt\""):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('value='):
                        split_l_2 = one.split('=')
                        ntt = int(clean_pattern.sub('', split_l_2[-1]))
            if line.startswith("<parameter name=\"nmc\""):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('value='):
                        split_l_2 = one.split('=')
                        nmc = int(clean_pattern.sub('', split_l_2[-1]))

            if line.startswith("<search_hit"):
                count+=1
                if count>1:
                    continue
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('peptide='):
                        split_l_2 = one.split('=')
                        original_pep = clean_pattern.sub('', split_l_2[-1])
                        identified_pep = original_pep
                        PTM_score=identified_pep.count('~')
                    if one.startswith("protein="):
                        split_l_2 = one.split('=')
                        protein_l.append(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith("calc_neutral_pep_mass="):
                        split_l_2 = one.split('=')
                        CalculatedParentMass=float(clean_pattern.sub('', split_l_2[-1]))
                    if one.startswith("massdiff="):
                        split_l_2 = one.split('=')
                        MassDiff=float(clean_pattern.sub('', split_l_2[-1]))

            if line.startswith("<modification_info modified_peptide"):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('modified_peptide='):
                        split_l_2 = one.split('=')
                        identified_pep = C_pattern.sub('C', (clean_pattern.sub('', split_l_2[-1])))
                        identified_pep = M_pattern.sub('M~', (clean_pattern.sub('', split_l_2[-1])))
            if line.startswith("<alternative_protein"):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('protein='):
                        split_l_2 = one.split('=')
                        tmp_str = clean_pattern.sub('', split_l_2[-1])
                        if tmp_str not in protein_l:
                            protein_l.append(tmp_str)
            if line.startswith("<peptideprophet_result "):
                split_l = line.split(' ')
                for one in split_l:
                    if one.startswith('probability='):
                        split_l_2 = one.split('=')
                        iProbability = float(clean_pattern.sub('', split_l_2[-1]))
            if line.startswith("</spectrum_query>"):
                PSM_Label = False
                for p in protein_l:
                    if 'Rev' not in p:
                        PSM_Label = True
                        break
                PSMs.append(PSM(filename,file_id,scan_id,charge_id,'NA',MeasuredParentMass, CalculatedParentMass, MassDiff,iProbability,PTM_score,identified_pep,PSM_Label,protein_l,','.join(protein_l),len(protein_l)))
                protein_l = []


    return PSMs



if __name__ == "__main__":
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:p:d:o:f:")
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Re-score results by WinnowNet\n")
        print("-p\t Original PSM file\n")
        print("-d\t Decoy Prefix used for target-decoy strategy\n")
        print("-o\t Prefix of filtered result at user-defined FDR at PSM and peptide level\n")
        print("-f\t FDR. Default: 0.01")
        sys.exit(1)
        start_time=time.time()
    input_file=""
    PSM_file=""
    filtered_prefix=""
    fdr=0.01
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Re-score results by WinnowNet\n")
            print("-p\t Original PSM file\n")
            print("-d\t Decoy Prefix used for target-decoy strategy\n")
            print("-o\t Prefix of filtered result at user-defined FDR at PSM and peptide level\n")
            print("-f\t FDR. Default: 0.01")
            sys.exit(1)
        elif opt in ("-i"):
            input_file=arg
        elif opt in ("-p"):
            PSM_file=arg
        elif opt in ("-d"):
            decoy_prefix=arg
        elif opt in ("-o"):
            filtered_prefix=arg
        elif opt in ("-f"):
            fdr=float(arg)

    PSMs = readWinnowNetData(PSM_file,input_file)
    psm_list = sorted(PSMs, key=lambda psm: (psm.rescore, psm.Massdiff, psm.PTM_score), reverse=True)
    rank_list = re_rank(PSMs)

    filter_list = show_Fdr(rank_list, fdr)
    print('psm:' + str(len(filter_list)))
    
    with open(filtered_prefix+'.psm.txt', 'w') as f:
        psm_out_list = ['Filename',  # 0
                    'ScanNumber',  # 1
                    'ParentCharge',  # 2
                    'MeasuredParentMass',  # 3
                    'CalculatedParentMass',  # 4
                    'MassErrorDa',  # 5 CalculatedParentMass - MeasuredParentMass
                    'MassErrorPPM',  # 6 MassErrorDa / CalculatedParentMass
                    'ScanType',  # 7
                    'SearchName',  # 8
                    'ScoringFunction',  # 9
                    'Score',  # 10
                    'DeltaZ',  # 11 the difference score between the rank 1 and 2
                    'DeltaP',  # 12
                    'IdentifiedPeptide',  # 13
                    'OriginalPeptide',  # 14
                    'ProteinNames',  # 15
                    'ProteinCount',  # 16
                    'TargetMatch']  # 17
        f.write('\t'.join(psm_out_list) + '\n')

            
        for psm in filter_list:
            TargetMatch = 'F'
            if psm.PSM_Label == True:
                TargetMatch = 'T'
            f.write(str(psm.filename)+ '\t' + str(psm.scan) + '\t' + str(psm.ParentCharge)+'\t'+str(psm.MeasuredParentMass)+'\t'+str(psm.CalculatedParentMass)+'\t'+str(psm.Massdiff)+'\t'+str(psm.MassErrorPPM)+'\t'+str(psm.ScanType)+'\t'+str(psm.SearchName)+'\t'+str(psm.ScoringFunction)+'\t'+str(psm.rescore)+'\t'+str(psm.DeltaZ)+'\t'+str(psm.DeltaP)+'\t'+ str(psm.IdentifiedPeptide) + '\t' +str(psm.OriginalPeptide)+'\t'+str(psm.Proteinname)+'\t'+ str(psm.ProteinCount)+'\t'+TargetMatch + '\n')


    filter_pep_list = show_Fdr_Pep(rank_list, fdr)
    print('pep:' + str(len(filter_pep_list)))
    
    with open(filtered_prefix+'.pep.txt', 'w') as f:
        pep_out_list = ['IdentifiedPeptide',    #0
                    'ParentCharge',         #1
                    'OriginalPeptide',      #2
                    'ProteinNames',         #3
                    'ProteinCount',         #4
                    'TargetMatch',          #5
                    'SpectralCount',        #6 number of PSMs matched to this peptide
                    'BestScore',            #7 the highest score of those PSMs
                    'PSMs',                 #8 a list of PSMs matched to this peptide. Use{Filename[ScanNumber],Filename[ScanNumber]} format
                    'ScanType',             #9
                    'SearchName']           #10
        f.write('\t'.join(pep_out_list) + '\n')
        for key, pep in filter_pep_list.items():
            f.write(pep.IdentifiedPeptide+'\t'+str(pep.ParentCharge)+'\t'+pep.OriginalPeptide+'\t'+'{'+pep.ProteinNames+'}'+'\t'+str(pep.ProteinCount)+'\t'+pep.TargetMatch+'\t'+str(pep.SpectralCount)+'\t'+str(pep.BestScore)+'\t'+','.join(pep.PSMs)+'\t'+pep.ScanType+'\t'+pep.SearchName+ '\n')




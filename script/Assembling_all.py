class peptide:

    def __init__(self):
        self.identified_pep = ""
        self.score=None
        self.qvalue = None
        self.peprob = None
        self.filetype = ""
        self.protein_list = []

class scan:

    def __init__(self):
        self.fidx = 0
        self.charge = None
        self.scan_number = 0
        self.pep_list = []
        self.unique_peplist = []

    def add_pep(self, pep):
        self.pep_list.append(pep)

    def write_out_str(self):
        pep_sort_list=sorted(self.pep_list,key=lambda pep:pep.peprob)
        rank=0
        results=[]
        peptide_dict={}
        pep_qv_dict={}
        for oPeptide in pep_sort_list:
            if oPeptide.identified_pep not in peptide_dict:
                peptide_dict[oPeptide.identified_pep]=[oPeptide.filetype]
            else:
                peptide_dict[oPeptide.identified_pep].append(oPeptide.filetype)
            if oPeptide.identified_pep not in pep_qv_dict:
                pep_qv_dict[oPeptide.identified_pep]=[oPeptide.peprob]
            else:
                pep_qv_dict[oPeptide.identified_pep].append(oPeptide.peprob)

        pep_set=set()
        for oPeptideID,oPeptide in enumerate(pep_sort_list):
            newPeptide=oPeptide
            if oPeptideID==0:
                newPeptide.peprob=min(pep_qv_dict[oPeptide.identified_pep])
            else:
                newPeptide.peprob=max(pep_qv_dict[oPeptide.identified_pep])

            newPeptide.filetype=','.join(peptide_dict[oPeptide.identified_pep])
            if newPeptide.identified_pep not in pep_set:
                self.unique_peplist.append(newPeptide)
                pep_set.add(newPeptide.identified_pep)
            else:
                continue
        


        '''
        pep_select_list=[]
        pep_select_list.append(pep_sort_list[0])
        pep_select_list[0].filetype=','.join(peptide_dict[pep_sort_list[0].identified_pep])
        if pep_sort_list[-1].identified_pep!=pep_select_list[0].identified_pep:
            pep_select_list.append(pep_sort_list[-1])
            pep_select_list[1].filetype=','.join(peptide_dict[pep_sort_list[-1].identified_pep])
        '''

        #for pep in pep_select_list:
        for pep in self.unique_peplist:
            rank+=1
            l=[]
            l.append(str(self.fidx)+'_'+str(self.scan_number)+'_'+str(self.charge)+'_'+str(rank))
            l.append(str(pep.peprob))
            l.append(pep.filetype)
            l.append(pep.identified_pep)
            l.append('\t'.join(pep.protein_list))
            results.append(l)
        return results


def read_comet_csv(input_file_str,psm_dict):
    print('Readin Comet-Percolator result')
    f=open(input_file_str)
    for line_id,line in enumerate(f):
        if line_id>0:
            line=line.replace('new_ms2/','')
            s=line.strip().split('\t')
            PSMId=s[0].strip().split('_')
            fileidx=str('_'.join(PSMId[:-3]))
            scannum=PSMId[-3]
            charge=PSMId[-2]
            score=float(s[1])
            qvalue=float(s[2])
            peprob=float(s[3])
            peptidestr=s[4].replace('[15.9949]','~')
            protein=s[5:len(s)]
            pep = peptide()
            pep.filetype='Comet'
            pep.score=score
            pep.qvalue=qvalue
            pep.peprob=peprob
            pep.identified_pep=peptidestr
            pep.protein_list=protein
            uniqueID=fileidx+'_'+scannum+'_'+charge
            if uniqueID in psm_dict.keys():
                psm_dict[uniqueID].add_pep(pep)
            else:
                one_scan=scan()
                one_scan.fidx=fileidx
                one_scan.scan_number=scannum
                one_scan.charge=charge
                one_scan.add_pep(pep)
                psm_dict[uniqueID]=one_scan

    f.close()
    print('Comet-Percolator finished!')

def read_myrimatch_csv(input_file_str,psm_dict):
    print('Readin Myrimatch-Percolator file')
    f=open(input_file_str)
    for line_id,line in enumerate(f):
        if line_id>0:
            s=line.strip().split('\t')
            PSMId=s[0].replace('.','').strip().split('_')
            fileidx=str('_'.join(PSMId[:-3]))
            scannum=PSMId[-3]
            charge=PSMId[-2]
            score=float(s[1])
            qvalue=float(s[2])
            peprob=float(s[3])
            peptidestr=s[4]
            protein=s[5:len(s)]
            pep = peptide()
            pep.filetype='Myrimatch'
            pep.score=score
            pep.qvalue=qvalue
            pep.peprob=peprob
            pep.identified_pep=peptidestr
            pep.protein_list=protein
            uniqueID=fileidx+'_'+scannum+'_'+charge
            if uniqueID in psm_dict.keys():
                psm_dict[uniqueID].add_pep(pep)
            else:
                one_scan=scan()
                one_scan.fidx=fileidx
                one_scan.scan_number=scannum
                one_scan.charge = charge
                one_scan.add_pep(pep)
                psm_dict[uniqueID]=one_scan

    f.close()
    print('Myrimatch-Percolator finished!')


def read_msgf_csv(input_file_str,psm_dict):
    print('Readin MSGFP-Percolator file')
    f=open(input_file_str)
    for line_id,line in enumerate(f):
        if line_id>0:
            s=line.strip().split('\t')
            PSMId=s[0].strip().split('_')
            fileidx=str('_'.join(PSMId[:-3]))
            scannum=PSMId[-3]
            charge=PSMId[-2]
            score=float(s[1])
            qvalue=float(s[2])
            peprob=float(s[3])
            peptidestr=s[4].replace('[+42','')
            peptidestr=s[4].replace('+16','~')
            protein=s[5:len(s)]
            pep = peptide()
            pep.filetype='MSGFP'
            pep.score=score
            pep.qvalue=qvalue
            pep.peprob=peprob
            pep.identified_pep=peptidestr
            pep.protein_list=protein
            uniqueID=fileidx+'_'+scannum+'_'+charge
            if uniqueID in psm_dict.keys():
                psm_dict[uniqueID].add_pep(pep)
            else:
                one_scan=scan()
                one_scan.fidx=fileidx
                one_scan.scan_number=scannum
                one_scan.charge = charge
                one_scan.add_pep(pep)
                psm_dict[uniqueID]=one_scan

    f.close()
    print('MSGFP-Percolator finished!')


def write_output(psm_dict, output_file_str):
    with open(output_file_str, 'w') as fw:
        header_str = 'PSMId\tPEP\tFileType\tPeptide\tProtein'
        fw.write(header_str)
        fw.write('\n')
        for key in psm_dict:
            v=psm_dict[key]
            if v.pep_list:
                ll = v.write_out_str()
                for l in ll:
                    fw.write('\t'.join(l))
                    fw.write('\n')

    print('write_output is done.')

if __name__ == '__main__':
    psm_dict = dict()
    read_comet_csv('CometMarine1_all.tsv',psm_dict)
    read_myrimatch_csv('MyrimatchMarine1_all.tsv',psm_dict)
    read_msgf_csv('MSGFPMarine1_all.tsv',psm_dict)
    output_file_str = 'AssemblingMarine1_all.tsv'
    write_output(psm_dict, output_file_str)

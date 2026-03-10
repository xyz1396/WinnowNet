import glob
import pickle
Comet_dict=dict()
MSGF_dict=dict()
Myrimatch_dict=dict()
with open('/media/fs0199/easystore1/Protein/DeepFilterV2/spectra_features/assembly_features_raw/train/comet.pin') as f:
	for line in f:
		s=line.strip().split('\t')
		a=s[0].replace('new_ms2/','').split('_')
		specID='_'.join(a[:-2])
		charge=a[-2]
		peptide=s[26].replace('[15.9949]','~')
		expmass=s[3]
		calmass=s[4]
		Mass=s[12]
		dM=str(float(calmass)-float(expmass))
		absdM=str(abs(float(calmass)-float(expmass)))
		peplen=s[13]
		enzInt=s[-6]
		chargeSet=[0,0,0]
		if int(charge)>2:
			chargeSet[2]=1
		else:
			chargeSet[int(charge)-1]=1

		feature=[Mass,dM,absdM,peplen,enzInt,str(chargeSet[0]),str(chargeSet[1]),str(chargeSet[2])]
		Comet_dict[specID+'_'+charge+'_'+peptide]=feature
		

with open('/media/fs0199/easystore1/Protein/DeepFilterV2/spectra_features/assembly_features_raw/train/MSGF.pin') as f:
	for line in f:
		s=line.strip().split('\t')
		a=s[0].split('_')
		specID='_'.join(a[:-2])
		charge=a[-2]
		peptide=s[17].replace('+16','~')
		expmass=str(float(s[3])-float(charge)*1.00784)
		calmass=str(float(s[4])-float(charge)*1.00784)
		Mass=str(float(s[6])-float(charge)*1.00784)
		dM=str((float(calmass)-float(expmass)))
		absdM=str(abs((float(calmass)-float(expmass))))
		peplen=str(len(peptide.split('.')[1])-peptide.split('.')[1].count('~'))
		enzInt=str(peptide.split('.')[1][:-1].count('K')+peptide.split('.')[1][:-1].count('R'))
		chargeSet=[0,0,0]
		if int(charge)>2:
			chargeSet[2]=1
		else:
			chargeSet[int(charge)-1]=1

		feature=[Mass,dM,absdM,peplen,enzInt,str(chargeSet[0]),str(chargeSet[1]),str(chargeSet[2])]
		MSGF_dict[specID+'_'+charge+'_'+peptide]=feature

with open('/media/fs0199/easystore1/Protein/DeepFilterV2/spectra_features/assembly_features_raw/train/myrimatch.pin') as f:
	for line in f:
		s=line.strip().split('\t')
		a=s[0].split('_')
		specID='_'.join(a[:-2]).replace('.','')
		charge=a[-2]
		peptide=s[18]
		expmass=s[3]
		calmass=s[4]
		Mass=str(float(expmass)+1.00784)
		dM=str(float(calmass)-float(expmass))
		absdM=str(abs(float(calmass)-float(expmass)))
		peplen=str(s[9])
		enzInt=str(peptide.split('.')[1][:-1].count('K')+peptide.split('.')[1][:-1].count('R'))
		chargeSet=[0,0,0]
		if int(charge)>2:
			chargeSet[2]=1
		else:
			chargeSet[int(charge)-1]=1

		feature=[Mass,dM,absdM,peplen,enzInt,str(chargeSet[0]),str(chargeSet[1]),str(chargeSet[2])]
		Myrimatch_dict[specID+'_'+charge+'_'+peptide]=feature


train_files=glob.glob('/media/fs0199/easystore1/Protein/DeepFilterV2/spectra_features/assembly_features_raw/PSMs/train/*tsv')
for file in train_files:
	psmfeature_dict=dict()
	with open(file) as f:
		for line in f:
			s=line.strip().split('\t')
			a=s[1].split('_')
			specID='_'.join(a[:-2])
			charge=a[-2]
			peptide=s[4]
			key=specID+'_'+charge+'_'+peptide
			if key in Comet_dict:
				psmfeature_dict[s[1]]=Comet_dict[key]
			elif key in Myrimatch_dict:
				psmfeature_dict[s[1]]=Myrimatch_dict[key]
			else:
				psmfeature_dict[s[1]]=MSGF_dict[key]

	writename=file.replace('/media/fs0199/easystore1/Protein/DeepFilterV2/spectra_features/assembly_features_raw/PSMs/train/','/media/fs0199/easystore1/Protein/DeepFilterV2/spectra_features/assembly_features_raw/train/').replace('tsv','pkl')
	with open(writename,'wb') as fw:
		pickle.dump(psmfeature_dict,fw)


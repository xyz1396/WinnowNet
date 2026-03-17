
```sh
python script/SpectraFeatures.py \
-i ../sipros5/sip_example/pct1/PanC_082924_07_ND1_filtered_psms.tsv \
-1 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT1 \
-2 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT2 \
-c script/SIP.cfg \
-o test/pct1.pkl \
-t 5 \
-f cnn \
-w 10

python script/SpectraFeatures.py \
-i ../sipros5/sip_example/pct1/PanC_082924_07_ND1_filtered_psms.tsv \
-1 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT1 \
-2 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT2 \
-c script/SIP.cfg \
-o test/pct1.pkl \
-t 3 \
-f att \
-w 10

python script/SpectraFeatures.py \
-i ../sipros5/sip_example/pct2/PanC_082924_08_ND2_filtered_psms.tsv \
-1 ../sipros5/sip_example/pct2/PanC_082924_08_ND2.FT1 \
-2 ../sipros5/sip_example/pct2/PanC_082924_08_ND2.FT2 \
-c script/SIP.cfg \
-o test/pct2.pkl \
-t 3 \
-f att \
-w 10

python3 test/check_features.py -i test/pct1.pkl --mode auto

python3 script/WinnowNet_Att.py \
  --target test/pct2.pkl \
  --decoy test/pct1.pkl \
  -m test/att.pt \
  --train-batch-size 64 \
  --eval-batch-size 128

```

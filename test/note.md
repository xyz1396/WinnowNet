
```sh
python script/SpectraFeatures.py \
-i ../sipros5/sip_example/pct1/PanC_082924_07_ND1_filtered_psms.tsv \
-1 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT1 \
-2 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT2 \
-c script/SIP.cfg \
-o test/pct1.pkl \
-t 5 \
-f att \
-w 10

python script/SpectraFeatures.py \
-i ../sipros5/sip_example/pct1/PanC_082924_07_ND1_filtered_psms.tsv \
-1 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT1 \
-2 ../sipros5/sip_example/pct1/PanC_082924_07_ND1.FT2 \
-c script/SIP.cfg \
-o test/pct1.pkl \
-t 5 \
-f cnn \
-w 10

```

echo '==> Running on Cora dataset.'
python -u ../train.py --data=cora --wd1=0.025603293518558053 --wd2=0.000004863005572141192 --wd3=0.0436590395673514 --tau=0 --layer=16 --hidden=32 --dropout=0.5242568305133181 --lr=0.01814417800980796 --seed=12648430
echo '==> Running on Pubmed dataset.'
python -u ../train.py --data=pubmed --wd1=0.025603293518558053 --wd2=0.000004863005572141192 --wd3=0.0436590395673514 --tau=0.16652287872123722 --layer=2 --hidden=32 --dropout=0.03986725760342571 --lr=0.024902524244662086 --seed=12648430
echo '==> Running on Citeseer dataset.'
python -u ../train.py --data=citeseer --wd1=0.025603293518558053 --wd2=0.000004863005572141192 --wd3=0.0436590395673514 --tau=0 --layer=16 --hidden=32 --dropout=0.5242568305133181 --lr=0.01814417800980796 --seed=12648430
echo '==> Running on Chameleon dataset.'
python -u ../train.py --data=chameleon --wd1=0.000023143469469968894 --wd2=0.00002608835674408066 --wd3=0.015592166693211333 --tau=0.35 --layer=16 --hidden=128 --dropout=0.5782208252722622 --lr=0.01113302773712752 --seed=12648430
echo '==> Running on Squirrel dataset.'
python -u ../train.py --data=squirrel --wd1=0.000023143469469968894 --wd2=0.00002608835674408066 --wd3=0.015592166693211333 --tau=0.35 --layer=16 --hidden=128 --dropout=0.5782208252722622 --lr=0.01113302773712752 --seed=12648430
echo '==> Running on Actor dataset.'
python -u ../train.py --data=film --wd1=0.0882469522859853 --wd2=0.00048203424854970474 --wd3=0.027915943550063556 --tau=0.35 --layer=4 --hidden=32 --dropout=0.05391883899763822 --lr=0.03117666983549838 --seed=12648430
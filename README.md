# image-classification-pytorch

## How to execute 

### Train
python main.py --task train --dataset data/ --model model/activity.pth --label-bin model/lb.pickle --epochs 25 --plot output/plot.png --num-classes 22


### Predict 
python main.py --task predict --model model/activity.pth --label-bin model/lb.pickle --image data/motogp/00000047.jpg --num-classes 22

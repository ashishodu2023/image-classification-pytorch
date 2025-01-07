# image-classification-pytorch

## How to execute code in local

### Train
python main.py --task train --dataset data/ --model model/activity.pth --label-bin model/lb.pickle --epochs 10 --plot output/plot.png --num-classes 22


### Predict 
python main.py --task predict --model model/activity.pth --label-bin model/lb.pickle --image data/motogp/00000047.jpg --num-classes 22

### MlFlow 
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
Visit http://127.0.0.1:5000


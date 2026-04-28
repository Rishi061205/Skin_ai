# Skin Disease AI 


## Quickstart

```INSTALL THE LIBRARIES 

pip install -r requirements.txt

```

python train.py       # trains the model, saves best checkpoint
python evaluate.py    # shows accuracy + confusion matrix
python predict.py --image path/to/skin.jpg   # test on one image

```

``` run the server

python app.py #flask api 
http://localhost:5000

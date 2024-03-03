# ShallowBKGC

The implementation of ShallowBKGC.

## Data

The three KGs we used as well as entity text information are in ./KGs.

## Reproducing results

## step1： Installing requirement packages
```shell
pip install -r requirements.txt
```
## step2：data preprocess
```shell
python Dataprocess.py
```
```shell
python Dataprocess_order.py
```
```shell
python Dataprocess_npy.py
```

#### step3：run the model on FB15k-237
```shell
python main_run_FB15k237.py
```

#### step4：run the model on WN18RR
```shell
python main_run_WN8RR.py
```

#### step5：run the model on YAGO3-10
```shell
python main_run_YAGO3-10.py
```
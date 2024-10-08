# Deep Learning Model

***Model Metrics Type: Binomial***<br/>
**Description: Metrics reported on full training frame**<br/>
 model id: rm-h2o-model-production_model-84<br/>
 frame id: rm-h2o-frame-production_model-84<br/>
 **MSE**: 0.13836128<br/>
 **RMSE**: 0.37196946<br/>
 **R^2**: 0.2902199<br/>
 **AUC**: 0.8416958<br/>
 **pr_auc**: 0.6486529<br/>
 **logloss**: 0.42375964<br/>
 **mean_per_class_error**: 0.24456012<br/>
 **default threshold**: 0.2931421101093292

# CM: Confusion Matrix

 ***(Row labels: Actual class; Column labels: Predicted class)***

```python
          	No   	Yes   	Error           Rate
    No  	3722   	935  	0.2008    	935 / 4,657
   Yes   	485  	1197  	0.2883    	485 / 1,682
Totals  	4207  	2132  	0.2240  	1,420 / 6,339
```

# Gains/Lift Table (Avg response rate: 26.53 %, avg score: 23.39 %):

```python
  Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate     Score  Cumulative Response Rate  Cumulative Score  Capture Rate  Cumulative Capture Rate        Gain  Cumulative Gain
      1                0.01009623         0.788777  3.297637         3.297637       0.875000  0.815366                  0.875000          0.815366      0.033294                 0.033294  229.763674       229.763674
      2                0.02003471         0.751855  3.110696         3.204902       0.825397  0.766515                  0.850394          0.791133      0.030916                 0.064209  211.069588       220.490230
      3                0.03013094         0.716027  2.767659         3.058392       0.734375  0.731988                  0.811518          0.771315      0.027943                 0.092152  176.765941       205.839159
      4                0.04006941         0.687884  3.290159         3.115877       0.873016  0.699631                  0.826772          0.753535      0.032699                 0.124851  229.015911       211.587724
      5                0.05000789         0.670042  3.230338         3.138625       0.857143  0.679258                  0.832808          0.738773      0.032105                 0.156956  223.033803       213.862497
      6                0.10001578         0.608147  2.591743         2.865184       0.687697  0.643602                  0.760252          0.691188      0.129608                 0.286564  159.174334       186.518415
      7                0.15002366         0.512866  2.116194         2.615521       0.561514  0.559793                  0.694006          0.647389      0.105826                 0.392390  111.619411       161.552080
      8                0.20003155         0.443216  1.949752         2.449079       0.517350  0.473503                  0.649842          0.603918      0.097503                 0.489893   94.975187       144.907857
      9                0.30004733         0.338072  1.706033         2.201397       0.452681  0.392882                  0.584122          0.533573      0.170630                 0.660523   70.603289       120.139668
     10                0.40006310         0.228267  1.224539         1.957182       0.324921  0.279174                  0.519322          0.469973      0.122473                 0.782996   22.453929        95.718233
     11                0.50007888         0.167897  0.915432         1.748832       0.242902  0.199116                  0.464038          0.415802      0.091558                 0.874554   -8.456772        74.883232
     12                0.59993690         0.099276  0.601329         1.557833       0.159558  0.132892                  0.413358          0.368712      0.060048                 0.934602  -39.867062        55.783328
     13                0.70026818         0.052537  0.361466         1.386423       0.095912  0.072873                  0.367876          0.326326      0.036266                 0.970868  -63.853398        38.642315
     14                0.79996845         0.018006  0.166969         1.234442       0.044304  0.034973                  0.327549          0.290014      0.016647                 0.987515  -83.303105        23.444226
     15                0.89998422         0.009060  0.089165         1.107167       0.023659  0.013359                  0.293777          0.259269      0.008918                 0.996433  -91.083452        10.716698
     16                1.00000000         0.002036  0.035666         1.000000       0.009464  0.005165                  0.265342          0.233855      0.003567                 1.000000  -96.433381         0.000000
```

**Status of Neuron Layers**
***(predicting Churn, 2-class classification, bernoulli distribution, CrossEntropy loss, 3,652 weights/biases, 47.8 KB, 63,390 training samples, mini-batch size 1)***

```python
 Layer Units      Type Dropout       L1       L2 Mean Rate Rate RMS Momentum Mean Weight Weight RMS Mean Bias Bias RMS
    1    19     Input  0.00 %
    2    50 Rectifier       0 0.000010 0.000000  0.427368 0.484791 0.000000   -0.032622   0.177904  0.237194 0.154306
    3    50 Rectifier       0 0.000010 0.000000  0.186815 0.270118 0.000000   -0.034294   0.141804  0.684008 0.235932
    4     2   Softmax         0.000010 0.000000  0.005937 0.006083 0.000000    0.042355   0.388837  0.000001 0.013630
```

**Scoring History**

```python
           Timestamp   Duration Training Speed   Epochs Iterations      Samples Training RMSE Training LogLoss Training r2 Training AUC Training pr_auc Training Lift Training Classification Error
 2024-08-13 21:32:51  0.000 sec                 0.00000          0     0.000000           NaN              NaN         NaN          NaN             NaN           NaN                           NaN
 2024-08-13 21:32:51  0.288 sec  24858 obs/sec  1.00000          1  6339.000000       0.40858          0.51821     0.14362      0.83497         0.63972       3.41541                       0.22291
 2024-08-13 21:32:51  0.528 sec  27206 obs/sec  2.00000          2 12678.000000       0.38271          0.44403     0.24863      0.83735         0.64003       3.41541                       0.23726
 2024-08-13 21:32:52  0.751 sec  28813 obs/sec  3.00000          3 19017.000000       0.37557          0.42999     0.27642      0.83532         0.63890       3.35652                       0.25035
 2024-08-13 21:32:52  0.961 sec  30185 obs/sec  4.00000          4 25356.000000       0.39657          0.47311     0.19324      0.83972         0.64684       3.35652                       0.23127
 2024-08-13 21:32:52  1.156 sec  31505 obs/sec  5.00000          5 31695.000000       0.37265          0.42467     0.28762      0.84076         0.64467       3.47430                       0.22874
 2024-08-13 21:32:52  1.334 sec  32872 obs/sec  6.00000          6 38034.000000       0.37386          0.43128     0.28298      0.84001         0.64652       3.23875                       0.24420
 2024-08-13 21:32:52  1.499 sec  34264 obs/sec  7.00000          7 44373.000000       0.37406          0.42698     0.28220      0.83598         0.63974       3.35652                       0.22732
 2024-08-13 21:32:52  1.655 sec  35562 obs/sec  8.00000          8 50712.000000       0.38002          0.44034     0.25918      0.84091         0.64576       3.41541                       0.25130
 2024-08-13 21:32:53  1.806 sec  36783 obs/sec  9.00000          9 57051.000000       0.37301          0.42558     0.28623      0.84241         0.64876       3.29764                       0.23805
 2024-08-13 21:32:53  1.962 sec  37958 obs/sec 10.00000         10 63390.000000       0.37197          0.42376     0.29022      0.84170         0.64865       3.29764                       0.22401
```

H2O version: 3.30.0.1-rm9.8.1

# Performances

```python
Criterion               	Value               	Standard Deviation
Accuracy		        0.7923607767613545	0.02148193519709103
Classification_error		0.20763922323864553	0.021481935197089817
AUC			        0.8314282009313267	0.023039517568668545
Precision		        0.7320460491889064	0.03092419337795061
Recall			        0.3798705966930266	0.062281138235821924
F_measure		        0.49809343740265133	0.058176392978831495
Sensitivity		        0.3798705966930266	0.062281138235821924
Specificity		        0.9480709724452469	0.00794791971729738
```

# Confusion Matrix

```python
	        true No	true Yes	class precision
pred. No	1387	342	        80.22%
pred. Yes	76	208	        73.24%
class recall	94.81%	37.82%
```

# Deep Learning-Lift Chart

![Screenshot 2024-08-13 230058](https://github.com/user-attachments/assets/33fa87af-61cc-48fb-891f-c9e87f0371ba)


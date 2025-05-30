## Run 1
- lr= 0.005
- arch= features->64->32->1
- optim= SGD
- epochs= 20
- Train Acc 80.06% | Val Acc 79.30% | Test Acc 79.47%

## Run 2
- lr= 0.001
- arch= features->64->32->1
- optim= SGD
- epochs= 20
- Train Acc 73.45% | Val Acc 73.46% | Test Acc 73.51%

## Run 3
- lr= 0.005
- arch= features->64->32->1
- optim= SGD
- epochs= 30
- added early stopping, patience= 3, min_delta=0.00
- Train Acc 80.62% | Val Acc 79.97% | Test Acc 79.09%

## Run 4
- lr= 0.005
- arch= features->64->32->1
- optim= SGD
- epochs= 50
- added early stopping, patience= 5, min_delta=0.00
- Train Acc 80.92% | Val Acc 80.13% | Test Acc 79.09%

## Run 5
- lr= 0.005
- arch= features->64->32->1
- optim= Adam
- epochs= 50
- added early stopping, patience= 5, min_delta= 0.001
- added dropout, 0.2 -> 0.1
- Train Acc 80.82% | Val Acc 77.46% | Test Acc 82.78%

## Run 6
- lr= 0.005
- arch= features->128->64->1
- optim= Adam
- epochs= 50
- added early stopping, patience= 5, min_delta= 0.001
- added dropout, 0.2
- Train Acc 82.11% | Val Acc 77.13% | Test Acc 80.04%

*most recent run with Run 6's architecture:* <br>
*Train Acc 81.57% | Val Acc 79.63% | Test Acc 79.00%*
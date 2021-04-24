import os

os.chdir(r'D:\WorkSpace\Python\Oxford\app\static\cnnimageretrieval-pytorch')
os.system(r'python -m cirtorch.examples.test --network-path retrievalSfM120k-resnet101-gem --datasets oxford5k --whitening retrieval-SfM-120k')

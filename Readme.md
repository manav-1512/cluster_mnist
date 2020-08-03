  UPDATE 
  CHECK cluster_new.py for latest code



MANAV SHAH
SE IT 60003180025
+91-9930447996 | manavhshah00@gmail.com

Assignment - 2:

The script files int the folder have the code for the operations performed on the dataset.

digit_cnn.py contains the code to build, train and save a Convolutional Neural Network Model that successfully predicts the digit a MINST image represents

cluster.py contains the code of preprocessing of images in the testSet directory, code for KMeans clustering applied to the image dataset and finding the accuracy of clustering the images representing a digit.

mnist.model has the trained CNN model with its achieved weights

PS C:\Users\HP\Desktop\MNIST>  & 'C:\Users\HP\AppData\Local\Programs\Python\Python38\python.exe' 'c:\Users\HP\.vscode\extensions\ms-python.python-2020.6.90262\pythonFiles\lib\python\debugpy\launcher' '58100' '--' 'c:\Users\HP\Desktop\MNIST\cluster.py'
Using TensorFlow backend.
2020-06-30 13:11:01.704881: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-06-30 13:11:01.813119: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-06-30 13:12:17.799251: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: 
nvcuda.dll not found
2020-06-30 13:12:17.813610: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)
2020-06-30 13:12:17.828235: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: JARVIS    
2020-06-30 13:12:17.840783: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: JARVIS
2020-06-30 13:12:17.852032: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-06-30 13:12:17.935680: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x13cbe4e47e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-30 13:12:17.968353: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Accuracy = 42.44 %

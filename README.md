# Writer Independent signature verification
Implemented a method to perform writer independent Signature verification. The model is trained on CEDAR dataset.
The data set contains 55 Original writers, each one provided 24 signatures. Each writer signature is forged 24 times.
Trained Siamese network with 50 original writers and tested on 5 test writers. It is able to identify with 100% accuracy all genuine vs forged signature with 100% accuracy.

# Framework
Python -Flask

Tensorflow


# To Run
1. Install required packages
2. Train the model by running sigverification.py
3. Use APIs for inference by running main.py

Two APIs are provided, one to onboard a new customer and other one to verify against existing customer.

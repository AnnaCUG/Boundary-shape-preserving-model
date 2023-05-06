#Boundary shape-preserving model for building mapping from high-resolution remote sensing images
This is the implementation for the paper "Boundary shape-preserving model for building mapping from high-resolution remote sensing images".
This is a building boundary extract method code for remote sensing images, and this is realized by Python. To run this project, you need to set up the environment, download the dataset, run a script to process data, and then you can train and test the network models. I will show you step by step to run this project, and I hope it is clear enough.

--Prerequisite I tested my project in Intel Core i9, 64G RAM, GPU RTX 3090. Because it takes several days for training, I recommend you use a CPU/GPU strong enough and about 24G Video Memory.

--Dataset I use a public remote sensing image, Spacenet, and WHU. The building footprints of the two datasets were stored in annotated files. The format of the footprints is the same as that of the COCO dataset, which includes coordinating information and object type.

--Training Run the following command python samples/coco/coco.py.

--Test Run the following command python sample/test.py.

When you encounter any problems during the reproduction process, you can leave a message asking me


# Active harmonic chain model의 active work fluctuation
The large deviation function encode the rarity of a certain observable. 
The problem is that the large deviation function is hard to obtain (intractable analytically).
However, the large deivation function for active work of active harmonic chain model can be obtained analytically.
So, We can use this model as a benchmark whether a large deiviation function estimator correctly estimate the LDF or not. 

I made two different neural networks. One is a fully-connected neural network which is already published in 2022 (Phys. Rev. E 105, 024115).
The other is permutation equivariant neural network which satisfy the particle permutation equivariance.
I claim that the permutation equivariant neural network is "correct" neural network which gives a correct information about the dynamics of the system.

You can run the codes main.py on your CPU and GPU. In Model.py file, the detailed neural network information is contained. 
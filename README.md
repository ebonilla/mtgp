# Multi-task Gaussian process (MTGP)
**Implements the multi-task model of Bonilla et al [1]**

## Author  
Edwin V. Bonilla

## Requirements
Yo need to have the gpml matlab package in your matlab path. You can download it at
http://gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v1.3-2006-09-08.tar.gz
Note that this is an old release of the gpml package as newer versions do not seem to support backward compatibility and MTGP was based on the version dated as 2006-03-29. 

## Main Contents
1. learn_mtgp.m : Learns an MTGP model. It uses the minimize function 
                  included in the gpml package.
2. nmargl_mtgp  : Marginal likelihood and its gradients for an MTGP model
2. alpha_mtgp.m :  Computes data structures for predictions in an MTGP model
3. predict_mtgp_all_tasks.m: Makes predictions for all tasks in an MTGP model
3. toy_example.m: A toy example of how to use the package 


## References
[1] Edwin V. Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams.
_Multi-task Gaussian Process Prediction_.
In Advances in Neural Information Processing Systems 20: NIPS'08.


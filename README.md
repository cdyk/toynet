# toynet - try to build a simple neural net from scratch

This is my neural net playground, where I try to build some neural nets completely from scratch, including deriving expressions mathematically. **Warning:** *Probably a lot of errors in here since I really don't know what I'm doing.*

## Simple feed-forward network on MNIST handwritten digits

Implementation in `tester.py`

Notation:
```
n0 - nodes in layer 0
n1 - nodes in layer 1
n2 - nodes in layer 2

      +-             -+       Weight matrix weighting nodes in 
      | w_11 ... w_1m |       layer 0 than is input to the
W_0 = |  ..       ..  |       activation func of layer 1
      | w_n1 ... w_nm |
      +-             -+       w_{output ix=row}{input ix=col}
```

Evaluation, from top to bottom:
```
o_0     output 0: layer of input nodes
s_0     sum 0: Applying W_0 on o_0 to form weighted sums
o_1     output 1: Output of activation func
s_1     sum 1: Applying W_1 on o_1
o_2     output 2: Output of activation func
E       error func of target t and o_2
```

Derivation of backpropagation, repeated application of the chain rule. Let
```
dx;dy      - partial derivative of x wrt y
G(x;y)     - gradient of x wrt y
J(x;y)     - Jacobian of x wrt y
diag(a..c) - Diagonal matrix with a ... c as entries
sigma_i    - Error propagated backwards until W_i
```
then
```
dE;dw_ij = G(e;o_2) J(o_2;w_ij)
         = G(e;o_2) J(o_2;s_1) J(s_1;w_ij)                       <- expr for W_1
         = G(e;o_2) J(o_2;s_1) J(s_1;o_1) J(o_1;w_ij)
         = G(e;o_2) J(o_2;s_1) J(s_1;o_1) J(o_1;s_0) J(s_1;w_ij) <- expr for W_0
          +---- sigma_1 -----+
          +----------- sigma_0 --------------------+
            1 x n2   n2 x n2    n2 x n1     n1 x n1
```

To find the search direction for the **last layer**, we use the sum of squares as error measure and get
```
           +-                        -+    +-                                -+
G(E;o_2) = | dE;do_2_1 ... dE;do_2_n2 | =  | (o_2_1 - t_1) ... (o_2_n3 - t_m) | 
           +-                        -+    +-                                -+
```
and the derivative of the activation function (we use the logistic func `phi(x)=1/(1+exp(-x)))` where `phi'(x) = phi(x)(1-phi(x))`))
```
J(o_2;s_1) = diag( do_2_1;ds_1_1 ... do_2_n2;ds_1_n2 )
           = diag( o_2_1(1 - o_2_1) ... o_2_n2(1 - o_2_n2) )
```
so `sigma_1` is the following `1 x n2`-matrix
```
                      +-                                                                       -+
G(e;o_2) J(o_2;s_1) = | (o_2_1 - t_1) o_2_1 (1 - o_2_1) ... (o_2_n3 - t_n2) o_2_n2 (1 - o_2_n2) |
                      +-                                                                       -+
```
Further, `J(s_1;w_ij)` is a `n2 x n1` matrix element where only `w_ij` is nonzero, so
```
sigma_1 J(s_1;w_ij) = ( 0 ... sigma_1_j o_1_ij ... 0)
```
the search direction for `W_1` is given by
```
         +-                                         -+
         | sigma_1_1 o_1_1   ...   sigma_1_n3 o_1_1  |
delta =  |      ..                        ..         |
         | sigma_1_1 o_1_n2  ...   sigma_1_n3 o_1_n2 |
         +-                                         -+
```

To find the search direction for the **hidden layer**, we continue build `sigma_0` from `sigma_1`
```
             +-                                -+
             | ds_1_1;do_1_1 ... ds_1_1;do_1_n1 |   s_1_i = < ( w_i1 ... w_im), (o_1_1, ... o_1_m) >
J(s_1;o_1) = |      ...               ...       |
             | ds_1_m;do_1_1 ... ds_1_m;do_1_n1 |   ds_1_i;do_1_j = w_ij 
             +-                                -+
             +-              -+
             | w_11  ... w_1m |
           = | ...       ...  | = W_1
             | w_n1  ... w_nm |
             +-              -+
J(o_1;s_0) = I( o_1_1;s_0_1 ... o_2_)
```
which gives the delta for the hidden layer.
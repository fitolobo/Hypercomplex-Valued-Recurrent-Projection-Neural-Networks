# Hypercomplex-Valued-Recurrent-Projection-Neural-Networks

Extension of the quaternion-valued recurrent projection neural network (QRPNNs) for processing hypercomplex-valued data, combining the projection storage method with the recurrent correlation neural networks (RCNNs).

## Usage

First of all, call the module using:

```
include("HyperRPNNs.jl")
```

### Bilinear Form and Activation Function

Effective use of hypercomplex-valued RPNN models requires a bilinear form and an appropriate hypercomplex-valued activation functions. Precisely, to take advantages of matrix operations, the hypercomplex-valued RPNN a function called BilinearForm whose inputs are a hypercomplex-valued matrix U organized in an array of size NxnxP, a hypercomplex-valued vector x organized in an array of size Nxn, and a set of parameters:
```
y = BilinearForm(U,x,Params)
```
Here, N denotes the the length of the vectors while n is the dimension of the hypercomplex numbers. For example, n=2 for complex numbers and n=4 for quaternions. The output y is such that y_i = \sum_{i=1}^N B(u_{ij},x_j), where B denotes the bilinar form. An example of the bilinear form is obtained using the command:
```
BilinearForm = HyperRPNNs.LambdaInner
```
In this function, the bilinear form coincides with a weighted inner product.

In a similar fashion, the activation function is defined as follows where x is a hypercomplex-valued vector organized in an array of size Nxn and ActFunctionParams is a set of parameters:
```
y = ActFunction(x,ActFunctionParams)
```
Examples of activation functions in the HyperRPNNs module include:
```
HyperRPNNs.csign, HyperRPNNs.twincsign, and HyperRPNNs.SplitSign
```
See the reference paper for details.

### Hypercomplex-Valued Recurrent Projection Neural Networks

The module contrains two different implementations of a hypercomplex-valued RPNN. One using synchronous update and the other using sequential (or asynchronous) update. Stability is ensured using both update modes. Synchronous and sequencial hypercomplex-valued RPNNs are called respectively using the commands:
```
y, Energy = HyperRPNNs.Sync(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
```
and
```
y, Energy = HyperRPNNs.Seq(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
```
Here, U is the hypercomplex-valued matrix whose columns correspond to the fundamental memories and organized in a real-valued array of size NxnxP (U[:,:,1] corresponds to the first fundamental memory) and xinput is a hypercomplex-valued vector organized in a real-valued array of size Nxn. The parameter alpha and beta define the excitation function f(x) = beta exp(alpha x). Finally, it_max specifies the maximum number of iterations.

See examples in the Jupyter notebook files.

## Authors
* **Marcos Eduardo Valle and Rodoldo Lobo** - *University of Campinas*


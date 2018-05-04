---
title: 'Notes on CoveNet'
date: 2015-12-20
permalink: /posts/2015/12/notes-on-cnn/
tags:
  - CNN
  - Deep Learning
  - Computer Vision
---

This is my study notes on the derivations for convolutional neural networks. If you have any question, please feel free to contact me by [zhongzisha@outlook.com](mailto:zhongzisha@outlook.com). Any comments are greatly appreciated!

Activation functions
--------------------

### sigmoid function

![\\begin{aligned}
f\\left(z\\right) & = & \\frac{1}{1+\\exp\\left(-z\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Af%5Cleft%28z%5Cright%29%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B1%2B%5Cexp%5Cleft%28-z%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
f\left(z\right) & = & \frac{1}{1+\exp\left(-z\right)}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial f}{\\partial z} & = & \\frac{\\exp\\left(-z\\right)}{\\left(1+\\exp\\left(-z\\right)\\right)\^{2}}=f\\left(z\\right)\\left(1-f\\left(z\\right)\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28-z%5Cright%29%7D%7B%5Cleft%281%2B%5Cexp%5Cleft%28-z%5Cright%29%5Cright%29%5E%7B2%7D%7D%3Df%5Cleft%28z%5Cright%29%5Cleft%281-f%5Cleft%28z%5Cright%29%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial f}{\partial z} & = & \frac{\exp\left(-z\right)}{\left(1+\exp\left(-z\right)\right)^{2}}=f\left(z\right)\left(1-f\left(z\right)\right)\end{aligned}")

### hyperbolic tangent function

![\\begin{aligned}
f\\left(z\\right) & = & A\\tanh\\left(Bz\\right)=1.7159\\tanh\\left(0.6666z\\right)=A\\frac{\\exp\\left(2Bz\\right)-1}{\\exp\\left(2Bz\\right)+1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Af%5Cleft%28z%5Cright%29%20%26%20%3D%20%26%20A%5Ctanh%5Cleft%28Bz%5Cright%29%3D1.7159%5Ctanh%5Cleft%280.6666z%5Cright%29%3DA%5Cfrac%7B%5Cexp%5Cleft%282Bz%5Cright%29-1%7D%7B%5Cexp%5Cleft%282Bz%5Cright%29%2B1%7D%5Cend%7Baligned%7D "\begin{aligned}
f\left(z\right) & = & A\tanh\left(Bz\right)=1.7159\tanh\left(0.6666z\right)=A\frac{\exp\left(2Bz\right)-1}{\exp\left(2Bz\right)+1}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial f}{\\partial z} & = & A\\frac{2B\\exp\\left(2Bz\\right)\\left(\\exp\\left(2Bz\\right)+1\\right)-\\left(\\exp\\left(2Bz\\right)-1\\right)2B\\exp\\left(2Bz\\right)}{\\left(\\exp\\left(2Bz\\right)+1\\right)\^{2}}=2AB\\frac{2\\exp\\left(2Bz\\right)}{\\left(\\exp\\left(2Bz\\right)+1\\right)\^{2}}=2B\\left(1-f\\left(z\\right)\^{2}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%20%26%20%3D%20%26%20A%5Cfrac%7B2B%5Cexp%5Cleft%282Bz%5Cright%29%5Cleft%28%5Cexp%5Cleft%282Bz%5Cright%29%2B1%5Cright%29-%5Cleft%28%5Cexp%5Cleft%282Bz%5Cright%29-1%5Cright%292B%5Cexp%5Cleft%282Bz%5Cright%29%7D%7B%5Cleft%28%5Cexp%5Cleft%282Bz%5Cright%29%2B1%5Cright%29%5E%7B2%7D%7D%3D2AB%5Cfrac%7B2%5Cexp%5Cleft%282Bz%5Cright%29%7D%7B%5Cleft%28%5Cexp%5Cleft%282Bz%5Cright%29%2B1%5Cright%29%5E%7B2%7D%7D%3D2B%5Cleft%281-f%5Cleft%28z%5Cright%29%5E%7B2%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial f}{\partial z} & = & A\frac{2B\exp\left(2Bz\right)\left(\exp\left(2Bz\right)+1\right)-\left(\exp\left(2Bz\right)-1\right)2B\exp\left(2Bz\right)}{\left(\exp\left(2Bz\right)+1\right)^{2}}=2AB\frac{2\exp\left(2Bz\right)}{\left(\exp\left(2Bz\right)+1\right)^{2}}=2B\left(1-f\left(z\right)^{2}\right)\end{aligned}")

### RELU function

![\\begin{aligned}
f\\left(z\\right) & = & \\max\\left(0,z\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Af%5Cleft%28z%5Cright%29%20%26%20%3D%20%26%20%5Cmax%5Cleft%280%2Cz%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
f\left(z\right) & = & \max\left(0,z\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial f}{\\partial z} & = & \\begin{cases}
0 & x&lt;0\\\\
1 & x&gt;0
\\end{cases}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%20%26%20%3D%20%26%20%5Cbegin%7Bcases%7D%0A0%20%26%20x%3C0%5C%5C%0A1%20%26%20x%3E0%0A%5Cend%7Bcases%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial f}{\partial z} & = & \begin{cases}
0 & x<0\\
1 & x>0
\end{cases}\end{aligned}")

### Leaky ReLU

![\\begin{aligned}
f\\left(z\\right) & = & \\begin{cases}
z & z\\ge0\\\\
\\frac{z}{a} & z&lt;0
\\end{cases}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Af%5Cleft%28z%5Cright%29%20%26%20%3D%20%26%20%5Cbegin%7Bcases%7D%0Az%20%26%20z%5Cge0%5C%5C%0A%5Cfrac%7Bz%7D%7Ba%7D%20%26%20z%3C0%0A%5Cend%7Bcases%7D%5Cend%7Baligned%7D "\begin{aligned}
f\left(z\right) & = & \begin{cases}
z & z\ge0\\
\frac{z}{a} & z<0
\end{cases}\end{aligned}")

where
![a\\in\\left(1,+\\infty\\right)](https://latex.codecogs.com/png.latex?a%5Cin%5Cleft%281%2C%2B%5Cinfty%5Cright%29 "a\in\left(1,+\infty\right)")

### Parametric ReLU

The ![a](https://latex.codecogs.com/png.latex?a "a") in Leaky ReLU is
learned in the training via backpropagation.

### Randomized Leaky ReLU

![\\begin{aligned}
f\\left(z\\right) & = & \\begin{cases}
z & z\\ge0\\\\
\\frac{z}{a} & z&lt;0
\\end{cases}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Af%5Cleft%28z%5Cright%29%20%26%20%3D%20%26%20%5Cbegin%7Bcases%7D%0Az%20%26%20z%5Cge0%5C%5C%0A%5Cfrac%7Bz%7D%7Ba%7D%20%26%20z%3C0%0A%5Cend%7Bcases%7D%5Cend%7Baligned%7D "\begin{aligned}
f\left(z\right) & = & \begin{cases}
z & z\ge0\\
\frac{z}{a} & z<0
\end{cases}\end{aligned}")

where
![a\\sim U\\left(l,u\\right)](https://latex.codecogs.com/png.latex?a%5Csim%20U%5Cleft%28l%2Cu%5Cright%29 "a\sim U\left(l,u\right)")
is sampled from a uniform distribution.

Neural Net
----------

We have a set of training samples:
![\\left\\{ x,y\\right\\} \_{i=1}\^{N}](https://latex.codecogs.com/png.latex?%5Cleft%5C%7B%20x%2Cy%5Cright%5C%7D%20_%7Bi%3D1%7D%5E%7BN%7D "\left\{ x,y\right\} _{i=1}^{N}"),
the neuron is defined as

![\\begin{aligned}
z\^{\\left(l+1\\right)} & = & W\^{\\left(l\\right)T}x\^{\\left(l\\right)}+b\^{\\left(l\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Az%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%20%26%20%3D%20%26%20W%5E%7B%5Cleft%28l%5Cright%29T%7Dx%5E%7B%5Cleft%28l%5Cright%29%7D%2Bb%5E%7B%5Cleft%28l%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
z^{\left(l+1\right)} & = & W^{\left(l\right)T}x^{\left(l\right)}+b^{\left(l\right)}\end{aligned}")

![x\^{\\left(l+1\\right)}=a\^{\\left(l+1\\right)}=f\\left(z\^{\\left(l+1\\right)}\\right)=\\frac{1}{1+e\^{-z\^{\\left(l+1\\right)}}}](https://latex.codecogs.com/png.latex?x%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%3Da%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%3Df%5Cleft%28z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cright%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7D "x^{\left(l+1\right)}=a^{\left(l+1\right)}=f\left(z^{\left(l+1\right)}\right)=\frac{1}{1+e^{-z^{\left(l+1\right)}}}")

where
![x\^{\\left(l\\right)}\\in R\^{n\_{l}\\times1}](https://latex.codecogs.com/png.latex?x%5E%7B%5Cleft%28l%5Cright%29%7D%5Cin%20R%5E%7Bn_%7Bl%7D%5Ctimes1%7D "x^{\left(l\right)}\in R^{n_{l}\times1}"),
![z\^{\\left(l+1\\right)}\\in R\^{n\_{l+1}\\times1}](https://latex.codecogs.com/png.latex?z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cin%20R%5E%7Bn_%7Bl%2B1%7D%5Ctimes1%7D "z^{\left(l+1\right)}\in R^{n_{l+1}\times1}"),
![b\^{\\left(l\\right)}\\in R\^{n\_{l+1}\\times1}](https://latex.codecogs.com/png.latex?b%5E%7B%5Cleft%28l%5Cright%29%7D%5Cin%20R%5E%7Bn_%7Bl%2B1%7D%5Ctimes1%7D "b^{\left(l\right)}\in R^{n_{l+1}\times1}"),
![W\^{\\left(l\\right)}\\in R\^{n\_{l}\\times n\_{l+1}}](https://latex.codecogs.com/png.latex?W%5E%7B%5Cleft%28l%5Cright%29%7D%5Cin%20R%5E%7Bn_%7Bl%7D%5Ctimes%20n_%7Bl%2B1%7D%7D "W^{\left(l\right)}\in R^{n_{l}\times n_{l+1}}").

then, we have

![\\begin{aligned}
\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial x\^{\\left(l\\right)}} & =\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial a\^{\\left(l\\right)}}= & W\^{\\left(l\\right)T},\\quad\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial W\^{\\left(l\\right)}}=x\^{\\left(l\\right)},\\quad\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial b\^{\\left(l\\right)}}=I\_{n\_{l+1}\\times n\_{l+1}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20x%5E%7B%5Cleft%28l%5Cright%29%7D%7D%20%26%20%3D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%20%26%20W%5E%7B%5Cleft%28l%5Cright%29T%7D%2C%5Cquad%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3Dx%5E%7B%5Cleft%28l%5Cright%29%7D%2C%5Cquad%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3DI_%7Bn_%7Bl%2B1%7D%5Ctimes%20n_%7Bl%2B1%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial z^{\left(l+1\right)}}{\partial x^{\left(l\right)}} & =\frac{\partial z^{\left(l+1\right)}}{\partial a^{\left(l\right)}}= & W^{\left(l\right)T},\quad\frac{\partial z^{\left(l+1\right)}}{\partial W^{\left(l\right)}}=x^{\left(l\right)},\quad\frac{\partial z^{\left(l+1\right)}}{\partial b^{\left(l\right)}}=I_{n_{l+1}\times n_{l+1}}\end{aligned}")

![\\frac{\\partial a\^{\\left(l+1\\right)}}{\\partial z\^{\\left(l+1\\right)}}=\\frac{\\partial f\\left(z\^{\\left(l+1\\right)}\\right)}{\\partial z\^{\\left(l+1\\right)}}=f'\\left(z\^{\\left(l+1\\right)}\\right)\\in R\^{n\_{l+1}\\times1}](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20a%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20f%5Cleft%28z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cright%29%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%3Df%27%5Cleft%28z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7Bl%2B1%7D%5Ctimes1%7D "\frac{\partial a^{\left(l+1\right)}}{\partial z^{\left(l+1\right)}}=\frac{\partial f\left(z^{\left(l+1\right)}\right)}{\partial z^{\left(l+1\right)}}=f'\left(z^{\left(l+1\right)}\right)\in R^{n_{l+1}\times1}")

Suppose we have an
![l\_{max}](https://latex.codecogs.com/png.latex?l_%7Bmax%7D "l_{max}")-layer
network. The loss function is defined as

![\\begin{aligned}
L & = & \\frac{1}{2}\\left|\\left|y-a\^{\\left(l\_{max}\\right)}\\right|\\right|\_{2}\^{2}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Cleft%7C%5Cleft%7Cy-a%5E%7B%5Cleft%28l_%7Bmax%7D%5Cright%29%7D%5Cright%7C%5Cright%7C_%7B2%7D%5E%7B2%7D%5Cend%7Baligned%7D "\begin{aligned}
L & = & \frac{1}{2}\left|\left|y-a^{\left(l_{max}\right)}\right|\right|_{2}^{2}\end{aligned}")

then,

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(l\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(l\\right)}}=\\begin{cases}
-\\left(y-a\^{\\left(l\_{max}\\right)}\\right) & l=l\_{max}\\\\
\\frac{\\partial L}{\\partial z\^{\\left(l+1\\right)}}\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial a\^{\\left(l\\right)}}=W\^{\\left(l\\right)}\\delta\^{\\left(z\^{l+1}\\right)} & otherwise
\\end{cases}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%28l%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%5Cbegin%7Bcases%7D%0A-%5Cleft%28y-a%5E%7B%5Cleft%28l_%7Bmax%7D%5Cright%29%7D%5Cright%29%20%26%20l%3Dl_%7Bmax%7D%5C%5C%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3DW%5E%7B%5Cleft%28l%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7Bl%2B1%7D%5Cright%29%7D%20%26%20otherwise%0A%5Cend%7Bcases%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(l\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(l\right)}}=\begin{cases}
-\left(y-a^{\left(l_{max}\right)}\right) & l=l_{max}\\
\frac{\partial L}{\partial z^{\left(l+1\right)}}\frac{\partial z^{\left(l+1\right)}}{\partial a^{\left(l\right)}}=W^{\left(l\right)}\delta^{\left(z^{l+1}\right)} & otherwise
\end{cases}\end{aligned}")

![\\begin{aligned}
\\delta\^{\\left(z\^{\\left(l\\right)}\\right)} & = & \\frac{\\partial L}{\\partial z\^{\\left(l\\right)}}=\\frac{\\partial L}{\\partial a\^{\\left(l\\right)}}\\frac{\\partial a\^{\\left(l\\right)}}{\\partial z\^{\\left(l\\right)}}=\\delta\^{\\left(a\^{\\left(l\\right)}\\right)}\\circ f'\\left(z\^{\\left(l+1\\right)}\\right)\\in R\^{n\_{l}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%28l%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%28l%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20a%5E%7B%5Cleft%28l%5Cright%29%7D%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%28l%5Cright%29%7D%5Cright%29%7D%5Ccirc%20f%27%5Cleft%28z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7Bl%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(z^{\left(l\right)}\right)} & = & \frac{\partial L}{\partial z^{\left(l\right)}}=\frac{\partial L}{\partial a^{\left(l\right)}}\frac{\partial a^{\left(l\right)}}{\partial z^{\left(l\right)}}=\delta^{\left(a^{\left(l\right)}\right)}\circ f'\left(z^{\left(l+1\right)}\right)\in R^{n_{l}\times1}\end{aligned}")

then

![\\begin{aligned}
\\nabla\_{W\^{\\left(l\\right)}}L & = & \\frac{\\partial L}{\\partial W\^{\\left(l\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(l+1\\right)}}\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial W\^{\\left(l\\right)}}=a\^{\\left(l\\right)}\\delta\^{\\left(z\^{\\left(l+1\\right)}\\right)T}\\in R\^{n\_{l}\\times n\_{l+1}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7BW%5E%7B%5Cleft%28l%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%28l%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cright%29T%7D%5Cin%20R%5E%7Bn_%7Bl%7D%5Ctimes%20n_%7Bl%2B1%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{W^{\left(l\right)}}L & = & \frac{\partial L}{\partial W^{\left(l\right)}}=\frac{\partial L}{\partial z^{\left(l+1\right)}}\frac{\partial z^{\left(l+1\right)}}{\partial W^{\left(l\right)}}=a^{\left(l\right)}\delta^{\left(z^{\left(l+1\right)}\right)T}\in R^{n_{l}\times n_{l+1}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{b\^{\\left(l\\right)}}L & = & \\frac{\\partial L}{\\partial b\^{\\left(l\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(l+1\\right)}}\\frac{\\partial z\^{\\left(l+1\\right)}}{\\partial b\^{\\left(l\\right)}}=\\delta\^{\\left(z\^{\\left(l+1\\right)}\\right)}\\in R\^{n\_{l+1}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bb%5E%7B%5Cleft%28l%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%28l%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%28l%2B1%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7Bl%2B1%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{b^{\left(l\right)}}L & = & \frac{\partial L}{\partial b^{\left(l\right)}}=\frac{\partial L}{\partial z^{\left(l+1\right)}}\frac{\partial z^{\left(l+1\right)}}{\partial b^{\left(l\right)}}=\delta^{\left(z^{\left(l+1\right)}\right)}\in R^{n_{l+1}\times1}\end{aligned}")

A Neural Net Case
-----------------

Suppose we have

![\\begin{aligned}
a\^{\\left(1\\right)} & = & x\\in R\^{n\_{1}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20x%5Cin%20R%5E%7Bn_%7B1%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(1\right)} & = & x\in R^{n_{1}\times1}\end{aligned}")

![z\^{\\left(2\\right)}=W\^{\\left(1\\right)T}a\^{\\left(1\\right)}+b\^{\\left(1\\right)},\\quad W\^{\\left(1\\right)}\\in R\^{n\_{1}\\times n\_{2}},b\^{\\left(1\\right)}\\in R\^{n\_{2}\\times1}](https://latex.codecogs.com/png.latex?z%5E%7B%5Cleft%282%5Cright%29%7D%3DW%5E%7B%5Cleft%281%5Cright%29T%7Da%5E%7B%5Cleft%281%5Cright%29%7D%2Bb%5E%7B%5Cleft%281%5Cright%29%7D%2C%5Cquad%20W%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B1%7D%5Ctimes%20n_%7B2%7D%7D%2Cb%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D "z^{\left(2\right)}=W^{\left(1\right)T}a^{\left(1\right)}+b^{\left(1\right)},\quad W^{\left(1\right)}\in R^{n_{1}\times n_{2}},b^{\left(1\right)}\in R^{n_{2}\times1}")

![a\^{\\left(2\\right)}=f\\left(z\^{\\left(2\\right)}\\right)\\in R\^{n\_{2}\\times1}](https://latex.codecogs.com/png.latex?a%5E%7B%5Cleft%282%5Cright%29%7D%3Df%5Cleft%28z%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D "a^{\left(2\right)}=f\left(z^{\left(2\right)}\right)\in R^{n_{2}\times1}")

![z\^{\\left(3\\right)}=W\^{\\left(2\\right)T}a\^{\\left(2\\right)}+b\^{\\left(2\\right)},\\quad W\^{\\left(2\\right)}\\in R\^{n\_{2}\\times n\_{3}},b\^{\\left(2\\right)}\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?z%5E%7B%5Cleft%283%5Cright%29%7D%3DW%5E%7B%5Cleft%282%5Cright%29T%7Da%5E%7B%5Cleft%282%5Cright%29%7D%2Bb%5E%7B%5Cleft%282%5Cright%29%7D%2C%5Cquad%20W%5E%7B%5Cleft%282%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes%20n_%7B3%7D%7D%2Cb%5E%7B%5Cleft%282%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "z^{\left(3\right)}=W^{\left(2\right)T}a^{\left(2\right)}+b^{\left(2\right)},\quad W^{\left(2\right)}\in R^{n_{2}\times n_{3}},b^{\left(2\right)}\in R^{n_{3}\times1}")

![a\^{\\left(3\\right)}=f\\left(z\^{\\left(3\\right)}\\right)\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?a%5E%7B%5Cleft%283%5Cright%29%7D%3Df%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "a^{\left(3\right)}=f\left(z^{\left(3\right)}\right)\in R^{n_{3}\times1}")

![z\^{\\left(4\\right)}=W\^{\\left(3\\right)T}a\^{\\left(3\\right)}+b\^{\\left(3\\right)},\\quad W\^{\\left(3\\right)}\\in R\^{n\_{3}\\times n\_{4}},b\^{\\left(3\\right)}\\in R\^{n\_{4}\\times1}](https://latex.codecogs.com/png.latex?z%5E%7B%5Cleft%284%5Cright%29%7D%3DW%5E%7B%5Cleft%283%5Cright%29T%7Da%5E%7B%5Cleft%283%5Cright%29%7D%2Bb%5E%7B%5Cleft%283%5Cright%29%7D%2C%5Cquad%20W%5E%7B%5Cleft%283%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes%20n_%7B4%7D%7D%2Cb%5E%7B%5Cleft%283%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D "z^{\left(4\right)}=W^{\left(3\right)T}a^{\left(3\right)}+b^{\left(3\right)},\quad W^{\left(3\right)}\in R^{n_{3}\times n_{4}},b^{\left(3\right)}\in R^{n_{4}\times1}")

![a\^{\\left(4\\right)}=f\\left(z\^{\\left(4\\right)}\\right)\\in R\^{n\_{4}\\times1}](https://latex.codecogs.com/png.latex?a%5E%7B%5Cleft%284%5Cright%29%7D%3Df%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D "a^{\left(4\right)}=f\left(z^{\left(4\right)}\right)\in R^{n_{4}\times1}")

For **Euclidean** loss:

![\\begin{aligned}
L & = & \\frac{1}{2}\\left|\\left|y-a\^{\\left(4\\right)}\\right|\\right|\_{2}\^{2},\\quad y\\in R\^{n\_{4}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Cleft%7C%5Cleft%7Cy-a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%7C%5Cright%7C_%7B2%7D%5E%7B2%7D%2C%5Cquad%20y%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
L & = & \frac{1}{2}\left|\left|y-a^{\left(4\right)}\right|\right|_{2}^{2},\quad y\in R^{n_{4}\times1}\end{aligned}")

For **cross-entropy** loss:

![\\begin{aligned}
a\_{i}\^{\\left(4\\right)} & = & \\frac{\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)}{\\sum\_{j=1}\^{n\_{4}}\\exp\\left(z\_{j}\^{\\left(4\\right)}\\right)}\\in R\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bn_%7B4%7D%7D%5Cexp%5Cleft%28z_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5Cend%7Baligned%7D "\begin{aligned}
a_{i}^{\left(4\right)} & = & \frac{\exp\left(z_{i}^{\left(4\right)}\right)}{\sum_{j=1}^{n_{4}}\exp\left(z_{j}^{\left(4\right)}\right)}\in R\end{aligned}")

![\\begin{aligned}
\\frac{\\partial a\_{i}\^{\\left(4\\right)}}{\\partial z\_{i}\^{\\left(4\\right)}} & = & \\frac{\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)\\sum\_{j=1}\^{n\_{4}}\\exp\\left(z\_{j}\^{\\left(4\\right)}\\right)-\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)}{\\left\\{ \\sum\_{j=1}\^{n\_{4}}\\exp\\left(z\_{j}\^{\\left(4\\right)}\\right)\\right\\} \^{2}}=a\_{i}\^{\\left(4\\right)}-a\_{i}\^{\\left(4\\right)}a\_{i}\^{\\left(4\\right)}=a\_{i}\^{\\left(4\\right)}\\left(1-a\_{i}\^{\\left(4\\right)}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Csum_%7Bj%3D1%7D%5E%7Bn_%7B4%7D%7D%5Cexp%5Cleft%28z_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29-%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%7B%5Cleft%5C%7B%20%5Csum_%7Bj%3D1%7D%5E%7Bn_%7B4%7D%7D%5Cexp%5Cleft%28z_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cright%5C%7D%20%5E%7B2%7D%7D%3Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%3Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial a_{i}^{\left(4\right)}}{\partial z_{i}^{\left(4\right)}} & = & \frac{\exp\left(z_{i}^{\left(4\right)}\right)\sum_{j=1}^{n_{4}}\exp\left(z_{j}^{\left(4\right)}\right)-\exp\left(z_{i}^{\left(4\right)}\right)\exp\left(z_{i}^{\left(4\right)}\right)}{\left\{ \sum_{j=1}^{n_{4}}\exp\left(z_{j}^{\left(4\right)}\right)\right\} ^{2}}=a_{i}^{\left(4\right)}-a_{i}^{\left(4\right)}a_{i}^{\left(4\right)}=a_{i}^{\left(4\right)}\left(1-a_{i}^{\left(4\right)}\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial a\_{i}\^{\\left(4\\right)}}{\\partial z\_{k}\^{\\left(4\\right)}} & = & \\frac{-\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)\\exp\\left\[\\left(z\_{k}\^{\\left(4\\right)}\\right)\\right\]}{\\left\\{ \\sum\_{j=1}\^{n\_{4}}\\exp\\left(z\_{j}\^{\\left(4\\right)}\\right)\\right\\} \^{2}}=-a\_{i}\^{\\left(4\\right)}a\_{k}\^{\\left(4\\right)},\\quad i\\neq k\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20z_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B-%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cexp%5Cleft%5B%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cright%5D%7D%7B%5Cleft%5C%7B%20%5Csum_%7Bj%3D1%7D%5E%7Bn_%7B4%7D%7D%5Cexp%5Cleft%28z_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cright%5C%7D%20%5E%7B2%7D%7D%3D-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7Da_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%2C%5Cquad%20i%5Cneq%20k%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial a_{i}^{\left(4\right)}}{\partial z_{k}^{\left(4\right)}} & = & \frac{-\exp\left(z_{i}^{\left(4\right)}\right)\exp\left[\left(z_{k}^{\left(4\right)}\right)\right]}{\left\{ \sum_{j=1}^{n_{4}}\exp\left(z_{j}^{\left(4\right)}\right)\right\} ^{2}}=-a_{i}^{\left(4\right)}a_{k}^{\left(4\right)},\quad i\neq k\end{aligned}")

![\\begin{aligned}
L & = & -\\sum\_{i=1}\^{C}\\log\\left\\{ \\left(a\_{i}\^{\\left(4\\right)}\\right)\^{y\_{i}}\\right\\} =-\\sum\_{i=1}\^{C}y\_{i}\\log\\left(a\_{i}\^{\\left(4\\right)}\\right)=-\\sum\_{j\\neq i}\^{C}y\_{j}\\log\\left(a\_{j}\^{\\left(4\\right)}\\right)-y\_{i}\\log\\left(a\_{i}\^{\\left(4\\right)}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20-%5Csum_%7Bi%3D1%7D%5E%7BC%7D%5Clog%5Cleft%5C%7B%20%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5E%7By_%7Bi%7D%7D%5Cright%5C%7D%20%3D-%5Csum_%7Bi%3D1%7D%5E%7BC%7Dy_%7Bi%7D%5Clog%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%3D-%5Csum_%7Bj%5Cneq%20i%7D%5E%7BC%7Dy_%7Bj%7D%5Clog%5Cleft%28a_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29-y_%7Bi%7D%5Clog%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
L & = & -\sum_{i=1}^{C}\log\left\{ \left(a_{i}^{\left(4\right)}\right)^{y_{i}}\right\} =-\sum_{i=1}^{C}y_{i}\log\left(a_{i}^{\left(4\right)}\right)=-\sum_{j\neq i}^{C}y_{j}\log\left(a_{j}^{\left(4\right)}\right)-y_{i}\log\left(a_{i}^{\left(4\right)}\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial a\_{i}\^{\\left(4\\right)}} & = & -y\_{i}\\frac{1}{a\_{i}\^{\\left(4\\right)}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20-y_%7Bi%7D%5Cfrac%7B1%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial a_{i}^{\left(4\right)}} & = & -y_{i}\frac{1}{a_{i}^{\left(4\right)}}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial z\_{i}\^{\\left(4\\right)}} & = & =-\\sum\_{j\\neq i}\^{C}y\_{j}\\frac{1}{a\_{j}\^{\\left(4\\right)}}\\left(-a\_{j}\^{\\left(4\\right)}a\_{i}\^{\\left(4\\right)}\\right)-y\_{i}\\frac{1}{a\_{i}\^{\\left(4\\right)}}a\_{i}\^{\\left(4\\right)}\\left(1-a\_{i}\^{\\left(4\\right)}\\right)=\\sum\_{j\\neq i}\^{C}y\_{j}a\_{i}\^{\\left(4\\right)}-y\_{i}\\left(1-a\_{i}\^{\\left(4\\right)}\\right)=\\sum\_{j=1}\^{C}y\_{j}a\_{i}\^{\\left(4\\right)}-y\_{i}=a\_{i}\^{\\left(4\\right)}-y\_{i}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20%3D-%5Csum_%7Bj%5Cneq%20i%7D%5E%7BC%7Dy_%7Bj%7D%5Cfrac%7B1%7D%7Ba_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cleft%28-a_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29-y_%7Bi%7D%5Cfrac%7B1%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%3D%5Csum_%7Bj%5Cneq%20i%7D%5E%7BC%7Dy_%7Bj%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D-y_%7Bi%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%3D%5Csum_%7Bj%3D1%7D%5E%7BC%7Dy_%7Bj%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D-y_%7Bi%7D%3Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D-y_%7Bi%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial z_{i}^{\left(4\right)}} & = & =-\sum_{j\neq i}^{C}y_{j}\frac{1}{a_{j}^{\left(4\right)}}\left(-a_{j}^{\left(4\right)}a_{i}^{\left(4\right)}\right)-y_{i}\frac{1}{a_{i}^{\left(4\right)}}a_{i}^{\left(4\right)}\left(1-a_{i}^{\left(4\right)}\right)=\sum_{j\neq i}^{C}y_{j}a_{i}^{\left(4\right)}-y_{i}\left(1-a_{i}^{\left(4\right)}\right)=\sum_{j=1}^{C}y_{j}a_{i}^{\left(4\right)}-y_{i}=a_{i}^{\left(4\right)}-y_{i}\end{aligned}")

then, we could get the error derivatives and the update rules as follows
(suppose we use Euclidean loss and sigmoid activation function):

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(4\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(4\\right)}}=-\\left(y-a\^{\\left(4\\right)}\\right)\\in R\^{n\_{4}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%284%5Cright%29%7D%7D%3D-%5Cleft%28y-a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(4\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(4\right)}}=-\left(y-a^{\left(4\right)}\right)\in R^{n_{4}\times1}\end{aligned}")

![\\delta\^{\\left(z\^{\\left(4\\right)}\\right)}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}=\\frac{\\partial L}{\\partial a\^{\\left(4\\right)}}\\frac{\\partial a\^{\\left(4\\right)}}{\\partial z\^{\\left(4\\right)}}=\\delta\^{\\left(a\^{\\left(4\\right)}\\right)}\\circ a\^{\\left(4\\right)}\\circ\\left(1-a\^{\\left(4\\right)}\\right)\\in R\^{n\_{4}\\times1}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20a%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Ccirc%20a%5E%7B%5Cleft%284%5Cright%29%7D%5Ccirc%5Cleft%281-a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D "\delta^{\left(z^{\left(4\right)}\right)}=\frac{\partial L}{\partial z^{\left(4\right)}}=\frac{\partial L}{\partial a^{\left(4\right)}}\frac{\partial a^{\left(4\right)}}{\partial z^{\left(4\right)}}=\delta^{\left(a^{\left(4\right)}\right)}\circ a^{\left(4\right)}\circ\left(1-a^{\left(4\right)}\right)\in R^{n_{4}\times1}")

![\\begin{aligned}
\\nabla\_{W\^{\\left(3\\right)}}L & = & \\frac{\\partial L}{\\partial W\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}\\frac{\\partial z\^{\\left(4\\right)}}{\\partial W\^{\\left(3\\right)}}=a\^{\\left(3\\right)}\\delta\^{\\left(z\^{\\left(4\\right)}\\right)T}\\in R\^{n\_{3}\\times n\_{4}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7BW%5E%7B%5Cleft%283%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%283%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%283%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29T%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes%20n_%7B4%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{W^{\left(3\right)}}L & = & \frac{\partial L}{\partial W^{\left(3\right)}}=\frac{\partial L}{\partial z^{\left(4\right)}}\frac{\partial z^{\left(4\right)}}{\partial W^{\left(3\right)}}=a^{\left(3\right)}\delta^{\left(z^{\left(4\right)}\right)T}\in R^{n_{3}\times n_{4}}\end{aligned}")

![\\nabla\_{b\^{\\left(3\\right)}}L=\\frac{\\partial L}{\\partial b\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}\\frac{\\partial z\^{\\left(4\\right)}}{\\partial b\^{\\left(3\\right)}}=\\delta\^{\\left(z\^{\\left(4\\right)}\\right)}\\in R\^{n\_{4}\\times1}](https://latex.codecogs.com/png.latex?%5Cnabla_%7Bb%5E%7B%5Cleft%283%5Cright%29%7D%7DL%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D "\nabla_{b^{\left(3\right)}}L=\frac{\partial L}{\partial b^{\left(3\right)}}=\frac{\partial L}{\partial z^{\left(4\right)}}\frac{\partial z^{\left(4\right)}}{\partial b^{\left(3\right)}}=\delta^{\left(z^{\left(4\right)}\right)}\in R^{n_{4}\times1}")

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(3\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}\\frac{\\partial z\^{\\left(4\\right)}}{\\partial a\^{\\left(3\\right)}}=W\^{\\left(3\\right)}\\delta\^{\\left(z\^{\\left(4\\right)}\\right)}\\in R\^{n\_{3}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%283%5Cright%29%7D%7D%3DW%5E%7B%5Cleft%283%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(3\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(3\right)}}=\frac{\partial L}{\partial z^{\left(4\right)}}\frac{\partial z^{\left(4\right)}}{\partial a^{\left(3\right)}}=W^{\left(3\right)}\delta^{\left(z^{\left(4\right)}\right)}\in R^{n_{3}\times1}\end{aligned}")

![\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial a\^{\\left(3\\right)}}\\frac{\\partial a\^{\\left(3\\right)}}{\\partial z\^{\\left(3\\right)}}=\\delta\^{\\left(a\^{\\left(3\\right)}\\right)}\\circ a\^{\\left(3\\right)}\\circ\\left(1-a\^{\\left(3\\right)}\\right)\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20a%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Ccirc%20a%5E%7B%5Cleft%283%5Cright%29%7D%5Ccirc%5Cleft%281-a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "\delta^{\left(z^{\left(3\right)}\right)}=\frac{\partial L}{\partial z^{\left(3\right)}}=\frac{\partial L}{\partial a^{\left(3\right)}}\frac{\partial a^{\left(3\right)}}{\partial z^{\left(3\right)}}=\delta^{\left(a^{\left(3\right)}\right)}\circ a^{\left(3\right)}\circ\left(1-a^{\left(3\right)}\right)\in R^{n_{3}\times1}")

![\\begin{aligned}
\\nabla\_{W\^{\\left(2\\right)}}L & = & \\frac{\\partial L}{\\partial W\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}\\frac{\\partial z\^{\\left(3\\right)}}{\\partial W\^{\\left(2\\right)}}=a\^{\\left(2\\right)}\\delta\^{\\left(z\^{\\left(3\\right)}\\right)T}\\in R\^{n\_{2}\\times n\_{3}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7BW%5E%7B%5Cleft%282%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%282%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%282%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29T%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes%20n_%7B3%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{W^{\left(2\right)}}L & = & \frac{\partial L}{\partial W^{\left(2\right)}}=\frac{\partial L}{\partial z^{\left(3\right)}}\frac{\partial z^{\left(3\right)}}{\partial W^{\left(2\right)}}=a^{\left(2\right)}\delta^{\left(z^{\left(3\right)}\right)T}\in R^{n_{2}\times n_{3}}\end{aligned}")

![\\nabla\_{b\^{\\left(2\\right)}}L=\\frac{\\partial L}{\\partial b\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}\\frac{\\partial z\^{\\left(3\\right)}}{\\partial b\^{\\left(2\\right)}}=\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?%5Cnabla_%7Bb%5E%7B%5Cleft%282%5Cright%29%7D%7DL%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "\nabla_{b^{\left(2\right)}}L=\frac{\partial L}{\partial b^{\left(2\right)}}=\frac{\partial L}{\partial z^{\left(3\right)}}\frac{\partial z^{\left(3\right)}}{\partial b^{\left(2\right)}}=\delta^{\left(z^{\left(3\right)}\right)}\in R^{n_{3}\times1}")

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(2\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}\\frac{\\partial z\^{\\left(3\\right)}}{\\partial a\^{\\left(2\\right)}}=W\^{\\left(2\\right)}\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}\\in R\^{n\_{2}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%3DW%5E%7B%5Cleft%282%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(2\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(2\right)}}=\frac{\partial L}{\partial z^{\left(3\right)}}\frac{\partial z^{\left(3\right)}}{\partial a^{\left(2\right)}}=W^{\left(2\right)}\delta^{\left(z^{\left(3\right)}\right)}\in R^{n_{2}\times1}\end{aligned}")

![\\delta\^{\\left(z\^{\\left(2\\right)}\\right)}=\\frac{\\partial L}{\\partial z\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial a\^{\\left(2\\right)}}\\frac{\\partial a\^{\\left(2\\right)}}{\\partial z\^{\\left(2\\right)}}=\\delta\^{\\left(a\^{\\left(2\\right)}\\right)}\\circ a\^{\\left(2\\right)}\\circ\\left(1-a\^{\\left(2\\right)}\\right)\\in R\^{n\_{2}\\times1}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Ccirc%20a%5E%7B%5Cleft%282%5Cright%29%7D%5Ccirc%5Cleft%281-a%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D "\delta^{\left(z^{\left(2\right)}\right)}=\frac{\partial L}{\partial z^{\left(2\right)}}=\frac{\partial L}{\partial a^{\left(2\right)}}\frac{\partial a^{\left(2\right)}}{\partial z^{\left(2\right)}}=\delta^{\left(a^{\left(2\right)}\right)}\circ a^{\left(2\right)}\circ\left(1-a^{\left(2\right)}\right)\in R^{n_{2}\times1}")

![\\begin{aligned}
\\nabla\_{W\^{\\left(1\\right)}}L & = & \\frac{\\partial L}{\\partial W\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(2\\right)}}\\frac{\\partial z\^{\\left(2\\right)}}{\\partial W\^{\\left(1\\right)}}=a\^{\\left(1\\right)}\\delta\^{\\left(z\^{\\left(2\\right)}\\right)T}\\in R\^{n\_{1}\\times n\_{2}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7BW%5E%7B%5Cleft%281%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%281%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%281%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29T%7D%5Cin%20R%5E%7Bn_%7B1%7D%5Ctimes%20n_%7B2%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{W^{\left(1\right)}}L & = & \frac{\partial L}{\partial W^{\left(1\right)}}=\frac{\partial L}{\partial z^{\left(2\right)}}\frac{\partial z^{\left(2\right)}}{\partial W^{\left(1\right)}}=a^{\left(1\right)}\delta^{\left(z^{\left(2\right)}\right)T}\in R^{n_{1}\times n_{2}}\end{aligned}")

![\\nabla\_{b\^{\\left(1\\right)}}L=\\frac{\\partial L}{\\partial b\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(2\\right)}}\\frac{\\partial z\^{\\left(2\\right)}}{\\partial b\^{\\left(1\\right)}}=\\delta\^{\\left(z\^{\\left(2\\right)}\\right)}\\in R\^{n\_{2}\\times1}](https://latex.codecogs.com/png.latex?%5Cnabla_%7Bb%5E%7B%5Cleft%281%5Cright%29%7D%7DL%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D "\nabla_{b^{\left(1\right)}}L=\frac{\partial L}{\partial b^{\left(1\right)}}=\frac{\partial L}{\partial z^{\left(2\right)}}\frac{\partial z^{\left(2\right)}}{\partial b^{\left(1\right)}}=\delta^{\left(z^{\left(2\right)}\right)}\in R^{n_{2}\times1}")

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(1\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(2\\right)}}\\frac{\\partial z\^{\\left(2\\right)}}{\\partial a\^{\\left(1\\right)}}=W\^{\\left(1\\right)}\\delta\^{\\left(z\^{\\left(2\\right)}\\right)}\\in R\^{n\_{1}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%281%5Cright%29%7D%7D%3DW%5E%7B%5Cleft%281%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B1%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(1\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(1\right)}}=\frac{\partial L}{\partial z^{\left(2\right)}}\frac{\partial z^{\left(2\right)}}{\partial a^{\left(1\right)}}=W^{\left(1\right)}\delta^{\left(z^{\left(2\right)}\right)}\in R^{n_{1}\times1}\end{aligned}")

![\\delta\^{\\left(z\^{\\left(1\\right)}\\right)}=\\frac{\\partial L}{\\partial z\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial a\^{\\left(2\\right)}}\\frac{\\partial a\^{\\left(2\\right)}}{\\partial z\^{\\left(1\\right)}}=\\delta\^{\\left(a\^{\\left(1\\right)}\\right)}\\circ a\^{\\left(1\\right)}\\circ\\left(1-a\^{\\left(1\\right)}\\right)\\in R\^{n\_{1}\\times1}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20z%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Ccirc%20a%5E%7B%5Cleft%281%5Cright%29%7D%5Ccirc%5Cleft%281-a%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B1%7D%5Ctimes1%7D "\delta^{\left(z^{\left(1\right)}\right)}=\frac{\partial L}{\partial z^{\left(1\right)}}=\frac{\partial L}{\partial a^{\left(2\right)}}\frac{\partial a^{\left(2\right)}}{\partial z^{\left(1\right)}}=\delta^{\left(a^{\left(1\right)}\right)}\circ a^{\left(1\right)}\circ\left(1-a^{\left(1\right)}\right)\in R^{n_{1}\times1}")

A CNN Case
----------

### I-C1-MP1-FC1-O

Suppose the net structure is I-C1-MP1-FC1-O, then we have

![\\begin{aligned}
a\^{\\left(1\\right)} & = & x\\in R\^{H\\times W\\times B}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20x%5Cin%20R%5E%7BH%5Ctimes%20W%5Ctimes%20B%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(1\right)} & = & x\in R^{H\times W\times B}\end{aligned}")

![zc\_{j}\^{\\left(1\\right)}=\\sum\_{i=1}\^{B}a\_{i}\^{\\left(1\\right)}\\star k\_{ij}\^{\\left(1\\right)}+b\_{j}\^{\\left(1\\right)}\\in R\^{\\left(H-h+1\\right)\\times\\left(W-w+1\\right)},\\quad j=1,\\cdots,F\_{1}](https://latex.codecogs.com/png.latex?zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BB%7Da_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cstar%20k_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%2Bb_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28H-h%2B1%5Cright%29%5Ctimes%5Cleft%28W-w%2B1%5Cright%29%7D%2C%5Cquad%20j%3D1%2C%5Ccdots%2CF_%7B1%7D "zc_{j}^{\left(1\right)}=\sum_{i=1}^{B}a_{i}^{\left(1\right)}\star k_{ij}^{\left(1\right)}+b_{j}^{\left(1\right)}\in R^{\left(H-h+1\right)\times\left(W-w+1\right)},\quad j=1,\cdots,F_{1}")

![ac\_{j}\^{\\left(1\\right)}=f\\left(zc\_{j}\^{\\left(1\\right)}\\right)\\in R\^{\\left(H-h+1\\right)\\times\\left(W-w+1\\right)}](https://latex.codecogs.com/png.latex?ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%3Df%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H-h%2B1%5Cright%29%5Ctimes%5Cleft%28W-w%2B1%5Cright%29%7D "ac_{j}^{\left(1\right)}=f\left(zc_{j}^{\left(1\right)}\right)\in R^{\left(H-h+1\right)\times\left(W-w+1\right)}")

![zp\_{j}\^{\\left(1\\right)}=maxpool(ac\_{j}\^{\\left(1\\right)},poolsize)\\in R\^{\\left(\\frac{H-h+1}{poolsize}\\right)\\times\\left(\\frac{W-w+1}{poolsize}\\right)}](https://latex.codecogs.com/png.latex?zp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%3Dmaxpool%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%2Cpoolsize%29%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH-h%2B1%7D%7Bpoolsize%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW-w%2B1%7D%7Bpoolsize%7D%5Cright%29%7D "zp_{j}^{\left(1\right)}=maxpool(ac_{j}^{\left(1\right)},poolsize)\in R^{\left(\frac{H-h+1}{poolsize}\right)\times\left(\frac{W-w+1}{poolsize}\right)}")

![ap\_{j}\^{\\left(1\\right)}=zp\_{j}\^{\\left(1\\right)}\\in R\^{\\left(\\frac{H-h+1}{poolsize}\\right)\\times\\left(\\frac{W-w+1}{poolsize}\\right)}](https://latex.codecogs.com/png.latex?ap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%3Dzp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH-h%2B1%7D%7Bpoolsize%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW-w%2B1%7D%7Bpoolsize%7D%5Cright%29%7D "ap_{j}^{\left(1\right)}=zp_{j}^{\left(1\right)}\in R^{\left(\frac{H-h+1}{poolsize}\right)\times\left(\frac{W-w+1}{poolsize}\right)}")

![\\begin{aligned}
a\^{\\left(2\\right)} & = & reshape(ap\_{j}\^{\\left(1\\right)})\\in R\^{n\_{2}\\times1},\\quad n\_{2}=\\left(\\frac{H-h+1}{poolsize}\\right)\\times\\left(\\frac{W-w+1}{poolsize}\\right)\\times F\_{1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%282%5Cright%29%7D%20%26%20%3D%20%26%20reshape%28ap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%29%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D%2C%5Cquad%20n_%7B2%7D%3D%5Cleft%28%5Cfrac%7BH-h%2B1%7D%7Bpoolsize%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW-w%2B1%7D%7Bpoolsize%7D%5Cright%29%5Ctimes%20F_%7B1%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(2\right)} & = & reshape(ap_{j}^{\left(1\right)})\in R^{n_{2}\times1},\quad n_{2}=\left(\frac{H-h+1}{poolsize}\right)\times\left(\frac{W-w+1}{poolsize}\right)\times F_{1}\end{aligned}")

![z\^{\\left(3\\right)}=W\^{\\left(2\\right)T}a\^{\\left(2\\right)}+b\^{\\left(2\\right)},\\quad W\^{\\left(2\\right)}\\in R\^{n\_{2}\\times n\_{3}},b\^{\\left(2\\right)}\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?z%5E%7B%5Cleft%283%5Cright%29%7D%3DW%5E%7B%5Cleft%282%5Cright%29T%7Da%5E%7B%5Cleft%282%5Cright%29%7D%2Bb%5E%7B%5Cleft%282%5Cright%29%7D%2C%5Cquad%20W%5E%7B%5Cleft%282%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes%20n_%7B3%7D%7D%2Cb%5E%7B%5Cleft%282%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "z^{\left(3\right)}=W^{\left(2\right)T}a^{\left(2\right)}+b^{\left(2\right)},\quad W^{\left(2\right)}\in R^{n_{2}\times n_{3}},b^{\left(2\right)}\in R^{n_{3}\times1}")

![a\^{\\left(3\\right)}=f\_{3}\\left(z\^{\\left(3\\right)}\\right)\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?a%5E%7B%5Cleft%283%5Cright%29%7D%3Df_%7B3%7D%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "a^{\left(3\right)}=f_{3}\left(z^{\left(3\right)}\right)\in R^{n_{3}\times1}")

For **Euclidean** loss:

![\\begin{aligned}
L & = & \\frac{1}{2}\\left|\\left|y-a\^{\\left(3\\right)}\\right|\\right|\_{2}\^{2}\\in R\^{n\_{3}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Cleft%7C%5Cleft%7Cy-a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%7C%5Cright%7C_%7B2%7D%5E%7B2%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
L & = & \frac{1}{2}\left|\left|y-a^{\left(3\right)}\right|\right|_{2}^{2}\in R^{n_{3}\times1}\end{aligned}")

For **cross-entropy** loss:

![\\begin{aligned}
a\_{i}\^{\\left(3\\right)} & = & \\frac{\\exp\\left(z\_{i}\^{\\left(3\\right)}\\right)}{\\sum\_{k}\\exp\\left(z\_{k}\^{\\left(3\\right)}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%7B%5Csum_%7Bk%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
a_{i}^{\left(3\right)} & = & \frac{\exp\left(z_{i}^{\left(3\right)}\right)}{\sum_{k}\exp\left(z_{k}^{\left(3\right)}\right)}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial a\_{i}\^{\\left(3\\right)}}{\\partial z\_{i}\^{\\left(3\\right)}} & = & \\frac{\\exp\\left(z\_{i}\^{\\left(3\\right)}\\right)\\sum\_{k}\\exp\\left(z\_{k}\^{\\left(3\\right)}\\right)-\\exp\\left(z\_{i}\^{\\left(3\\right)}\\right)\\exp\\left(z\_{i}\^{\\left(3\\right)}\\right)}{\\left(\\sum\_{k}\\exp\\left(z\_{k}\^{\\left(3\\right)}\\right)\\right)\^{2}}=a\_{i}\^{\\left(3\\right)}\\left(1-a\_{i}\^{\\left(3\\right)}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Csum_%7Bk%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29-%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%7B%5Cleft%28%5Csum_%7Bk%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cright%29%5E%7B2%7D%7D%3Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial a_{i}^{\left(3\right)}}{\partial z_{i}^{\left(3\right)}} & = & \frac{\exp\left(z_{i}^{\left(3\right)}\right)\sum_{k}\exp\left(z_{k}^{\left(3\right)}\right)-\exp\left(z_{i}^{\left(3\right)}\right)\exp\left(z_{i}^{\left(3\right)}\right)}{\left(\sum_{k}\exp\left(z_{k}^{\left(3\right)}\right)\right)^{2}}=a_{i}^{\left(3\right)}\left(1-a_{i}^{\left(3\right)}\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial a\_{j}\^{\\left(3\\right)}}{\\partial z\_{i}\^{\\left(3\\right)}} & = & \\frac{-\\exp\\left(z\_{j}\^{\\left(3\\right)}\\right)\\exp\\left(z\_{i}\^{\\left(3\\right)}\\right)}{\\left(\\sum\_{k}\\exp\\left(z\_{k}\^{\\left(3\\right)}\\right)\\right)\^{2}}=-a\_{j}\^{\\left(3\\right)}a\_{i}\^{\\left(3\\right)},\\quad i\\neq j\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20a_%7Bj%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B-%5Cexp%5Cleft%28z_%7Bj%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%7B%5Cleft%28%5Csum_%7Bk%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cright%29%5E%7B2%7D%7D%3D-a_%7Bj%7D%5E%7B%5Cleft%283%5Cright%29%7Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%2C%5Cquad%20i%5Cneq%20j%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial a_{j}^{\left(3\right)}}{\partial z_{i}^{\left(3\right)}} & = & \frac{-\exp\left(z_{j}^{\left(3\right)}\right)\exp\left(z_{i}^{\left(3\right)}\right)}{\left(\sum_{k}\exp\left(z_{k}^{\left(3\right)}\right)\right)^{2}}=-a_{j}^{\left(3\right)}a_{i}^{\left(3\right)},\quad i\neq j\end{aligned}")

![\\begin{aligned}
L & = & -\\sum\_{i=1}\^{C}y\_{i}\\log\\left(a\_{i}\^{\\left(3\\right)}\\right)=-\\sum\_{j\\neq i}y\_{j}\\log\\left(a\_{j}\^{\\left(3\\right)}\\right)-y\_{i}\\log\\left(a\_{i}\^{\\left(3\\right)}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20-%5Csum_%7Bi%3D1%7D%5E%7BC%7Dy_%7Bi%7D%5Clog%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%3D-%5Csum_%7Bj%5Cneq%20i%7Dy_%7Bj%7D%5Clog%5Cleft%28a_%7Bj%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29-y_%7Bi%7D%5Clog%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
L & = & -\sum_{i=1}^{C}y_{i}\log\left(a_{i}^{\left(3\right)}\right)=-\sum_{j\neq i}y_{j}\log\left(a_{j}^{\left(3\right)}\right)-y_{i}\log\left(a_{i}^{\left(3\right)}\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial a\_{i}\^{\\left(3\\right)}} & = & -y\_{i}\\frac{1}{a\_{i}\^{\\left(3\\right)}}\\\\
\\Rightarrow\\delta\^{\\left(a\^{\\left(3\\right)}\\right)} & = & -\\frac{y}{a\^{\\left(3\\right)}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%20%26%20%3D%20%26%20-y_%7Bi%7D%5Cfrac%7B1%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%5C%5C%0A%5CRightarrow%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20-%5Cfrac%7By%7D%7Ba%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial a_{i}^{\left(3\right)}} & = & -y_{i}\frac{1}{a_{i}^{\left(3\right)}}\\
\Rightarrow\delta^{\left(a^{\left(3\right)}\right)} & = & -\frac{y}{a^{\left(3\right)}}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial a\_{i}\^{\\left(3\\right)}} & = & -y\_{i}\\frac{1}{a\_{i}\^{\\left(3\\right)}}\\Rightarrow\\delta\^{\\left(a\^{\\left(3\\right)}\\right)}=-\\frac{y}{a\^{\\left(3\\right)}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%20%26%20%3D%20%26%20-y_%7Bi%7D%5Cfrac%7B1%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%5CRightarrow%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%3D-%5Cfrac%7By%7D%7Ba%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial a_{i}^{\left(3\right)}} & = & -y_{i}\frac{1}{a_{i}^{\left(3\right)}}\Rightarrow\delta^{\left(a^{\left(3\right)}\right)}=-\frac{y}{a^{\left(3\right)}}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial z\_{i}\^{\\left(3\\right)}} & = & -\\sum\_{j\\neq i}y\_{j}\\frac{1}{a\_{j}\^{\\left(3\\right)}}\\left(-a\_{j}\^{\\left(3\\right)}a\_{i}\^{\\left(3\\right)}\\right)-y\_{i}\\frac{1}{a\_{i}\^{\\left(3\\right)}}a\_{i}\^{\\left(3\\right)}\\left(1-a\_{i}\^{\\left(3\\right)}\\right)=\\sum\_{j\\neq i}y\_{j}a\_{i}\^{\\left(3\\right)}-y\_{i}+y\_{i}a\_{i}\^{\\left(3\\right)}=a\_{i}\^{\\left(3\\right)}-y\_{i}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%20%26%20%3D%20%26%20-%5Csum_%7Bj%5Cneq%20i%7Dy_%7Bj%7D%5Cfrac%7B1%7D%7Ba_%7Bj%7D%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cleft%28-a_%7Bj%7D%5E%7B%5Cleft%283%5Cright%29%7Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29-y_%7Bi%7D%5Cfrac%7B1%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%7Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%3D%5Csum_%7Bj%5Cneq%20i%7Dy_%7Bj%7Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D-y_%7Bi%7D%2By_%7Bi%7Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D%3Da_%7Bi%7D%5E%7B%5Cleft%283%5Cright%29%7D-y_%7Bi%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial z_{i}^{\left(3\right)}} & = & -\sum_{j\neq i}y_{j}\frac{1}{a_{j}^{\left(3\right)}}\left(-a_{j}^{\left(3\right)}a_{i}^{\left(3\right)}\right)-y_{i}\frac{1}{a_{i}^{\left(3\right)}}a_{i}^{\left(3\right)}\left(1-a_{i}^{\left(3\right)}\right)=\sum_{j\neq i}y_{j}a_{i}^{\left(3\right)}-y_{i}+y_{i}a_{i}^{\left(3\right)}=a_{i}^{\left(3\right)}-y_{i}\end{aligned}")

![\\begin{aligned}
\\Rightarrow\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}} & = & a\^{\\left(3\\right)}-y\\in R\^{n\_{3}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5CRightarrow%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%20%26%20%3D%20%26%20a%5E%7B%5Cleft%283%5Cright%29%7D-y%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\Rightarrow\delta^{\left(z^{\left(3\right)}\right)}=\frac{\partial L}{\partial z^{\left(3\right)}} & = & a^{\left(3\right)}-y\in R^{n_{3}\times1}\end{aligned}")

![\\begin{aligned}
\\nabla\_{W\^{\\left(2\\right)}}L & = & \\frac{\\partial L}{\\partial W\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}\\frac{\\partial z\^{\\left(3\\right)}}{\\partial W\^{\\left(2\\right)}}=a\^{\\left(2\\right)}\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}\\in R\^{n\_{2}\\times n\_{3}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7BW%5E%7B%5Cleft%282%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%282%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%282%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes%20n_%7B3%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{W^{\left(2\right)}}L & = & \frac{\partial L}{\partial W^{\left(2\right)}}=\frac{\partial L}{\partial z^{\left(3\right)}}\frac{\partial z^{\left(3\right)}}{\partial W^{\left(2\right)}}=a^{\left(2\right)}\delta^{\left(z^{\left(3\right)}\right)}\in R^{n_{2}\times n_{3}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{b\^{\\left(2\\right)}}L & = & \\frac{\\partial L}{\\partial b\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}\\frac{\\partial z\^{\\left(3\\right)}}{\\partial b\^{\\left(2\\right)}}=\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}\\in R\^{n\_{3}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bb%5E%7B%5Cleft%282%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{b^{\left(2\right)}}L & = & \frac{\partial L}{\partial b^{\left(2\right)}}=\frac{\partial L}{\partial z^{\left(3\right)}}\frac{\partial z^{\left(3\right)}}{\partial b^{\left(2\right)}}=\delta^{\left(z^{\left(3\right)}\right)}\in R^{n_{3}\times1}\end{aligned}")

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(2\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(3\\right)}}\\frac{\\partial z\^{\\left(3\\right)}}{\\partial a\^{\\left(2\\right)}}=W\^{\\left(2\\right)}\\delta\^{\\left(z\^{\\left(3\\right)}\\right)}\\in R\^{n\_{2}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%283%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%282%5Cright%29%7D%7D%3DW%5E%7B%5Cleft%282%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(2\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(2\right)}}=\frac{\partial L}{\partial z^{\left(3\right)}}\frac{\partial z^{\left(3\right)}}{\partial a^{\left(2\right)}}=W^{\left(2\right)}\delta^{\left(z^{\left(3\right)}\right)}\in R^{n_{2}\times1}\end{aligned}")

From the
![\\delta\^{\\left(a\^{\\left(2\\right)}\\right)}\\in R\^{n\_{2}\\times1}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B2%7D%5Ctimes1%7D "\delta^{\left(a^{\left(2\right)}\right)}\in R^{n_{2}\times1}"),
we could get
![\\delta\^{\\left(ap\_{j}\^{\\left(1\\right)}\\right)}\\in R\^{\\left(\\frac{H-h+1}{poolsize}\\right)\\times\\left(\\frac{W-w+1}{poolsize}\\right)}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28ap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH-h%2B1%7D%7Bpoolsize%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW-w%2B1%7D%7Bpoolsize%7D%5Cright%29%7D "\delta^{\left(ap_{j}^{\left(1\right)}\right)}\in R^{\left(\frac{H-h+1}{poolsize}\right)\times\left(\frac{W-w+1}{poolsize}\right)}")

Then, we have

![\\begin{aligned}
\\delta\^{\\left(zp\_{j}\^{\\left(1\\right)}\\right)} & = & \\delta\^{\\left(ap\_{j}\^{\\left(1\\right)}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28zp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cdelta%5E%7B%5Cleft%28ap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(zp_{j}^{\left(1\right)}\right)} & = & \delta^{\left(ap_{j}^{\left(1\right)}\right)}\end{aligned}")

then, we upsample the error sensitity and get

![\\begin{aligned}
\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)} & = & up\\left(\\delta\^{\\left(zp\_{j}\^{\\left(1\\right)}\\right)}\\right)\\in R\^{\\left(H-h+1\\right)\\times\\left(W-w+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20up%5Cleft%28%5Cdelta%5E%7B%5Cleft%28zp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H-h%2B1%5Cright%29%5Ctimes%5Cleft%28W-w%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(ac_{j}^{\left(1\right)}\right)} & = & up\left(\delta^{\left(zp_{j}^{\left(1\right)}\right)}\right)\in R^{\left(H-h+1\right)\times\left(W-w+1\right)}\end{aligned}")

then, we get

![\\begin{aligned}
\\delta\^{\\left(zc\_{j}\^{\\left(1\\right)}\\right)} & = & \\frac{\\partial L}{\\partial zc\_{j}\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial ac\_{j}\^{\\left(1\\right)}}\\frac{\\partial ac\_{j}\^{\\left(1\\right)}}{\\partial zc\_{j}\^{\\left(1\\right)}}=\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)}\\circ ac\_{j}\^{\\left(1\\right)}\\circ\\left(1-ac\_{j}\^{\\left(1\\right)}\\right)\\in R\^{\\left(H-h+1\\right)\\times\\left(W-w+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Ccirc%20ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Ccirc%5Cleft%281-ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H-h%2B1%5Cright%29%5Ctimes%5Cleft%28W-w%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(zc_{j}^{\left(1\right)}\right)} & = & \frac{\partial L}{\partial zc_{j}^{\left(1\right)}}=\frac{\partial L}{\partial ac_{j}^{\left(1\right)}}\frac{\partial ac_{j}^{\left(1\right)}}{\partial zc_{j}^{\left(1\right)}}=\delta^{\left(ac_{j}^{\left(1\right)}\right)}\circ ac_{j}^{\left(1\right)}\circ\left(1-ac_{j}^{\left(1\right)}\right)\in R^{\left(H-h+1\right)\times\left(W-w+1\right)}\end{aligned}")

then, we get the following gradients

![\\begin{aligned}
\\nabla\_{k\_{ij}\^{\\left(1\\right)}}L & = & \\frac{\\partial L}{\\partial zc\_{j}\^{\\left(1\\right)}}\\frac{\\partial zc\_{j}\^{\\left(1\\right)}}{\\partial k\_{ij}\^{\\left(1\\right)}}=rot180\\left(conv2\\left(a\_{i}\^{\\left(1\\right)},rot180\\left(\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)}\\right),'valid'\\right)\\right)\\in R\^{h\\times w}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bk_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%7B%5Cpartial%20k_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3Drot180%5Cleft%28conv2%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%2Crot180%5Cleft%28%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cright%29%2C%27valid%27%5Cright%29%5Cright%29%5Cin%20R%5E%7Bh%5Ctimes%20w%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{k_{ij}^{\left(1\right)}}L & = & \frac{\partial L}{\partial zc_{j}^{\left(1\right)}}\frac{\partial zc_{j}^{\left(1\right)}}{\partial k_{ij}^{\left(1\right)}}=rot180\left(conv2\left(a_{i}^{\left(1\right)},rot180\left(\delta^{\left(ac_{j}^{\left(1\right)}\right)}\right),'valid'\right)\right)\in R^{h\times w}\end{aligned}")

![\\begin{aligned}
\\nabla\_{b\_{j}\^{\\left(1\\right)}}L & = & \\frac{\\partial L}{\\partial zc\_{j}\^{\\left(1\\right)}}\\frac{\\partial ac\_{j}\^{\\left(1\\right)}}{\\partial b\_{j}\^{\\left(1\\right)}}=\\sum\_{u,v}\\left(\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)}\\right)\_{u,v}\\in R\^{1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bb_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Csum_%7Bu%2Cv%7D%5Cleft%28%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cright%29_%7Bu%2Cv%7D%5Cin%20R%5E%7B1%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{b_{j}^{\left(1\right)}}L & = & \frac{\partial L}{\partial zc_{j}^{\left(1\right)}}\frac{\partial ac_{j}^{\left(1\right)}}{\partial b_{j}^{\left(1\right)}}=\sum_{u,v}\left(\delta^{\left(ac_{j}^{\left(1\right)}\right)}\right)_{u,v}\in R^{1}\end{aligned}")

### I-C1-MP1-C2-MP2-FC1-O

Suppose the net structure is I-C1-MP1-C2-MP2-FC1-O. For I1-C1-MP1, we
have
![j=1,\\cdots,F\_{1}](https://latex.codecogs.com/png.latex?j%3D1%2C%5Ccdots%2CF_%7B1%7D "j=1,\cdots,F_{1}")
where ![F\_{1}](https://latex.codecogs.com/png.latex?F_%7B1%7D "F_{1}")
is the number of convolution feature maps, and

![\\begin{aligned}
a\^{\\left(1\\right)} & = & x\\in R\^{H\_{1}\\times W\_{1}\\times B\_{1}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20x%5Cin%20R%5E%7BH_%7B1%7D%5Ctimes%20W_%7B1%7D%5Ctimes%20B_%7B1%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(1\right)} & = & x\in R^{H_{1}\times W_{1}\times B_{1}}\end{aligned}")

![\\begin{aligned}
zc\_{j}\^{\\left(1\\right)} & = & \\sum\_{i=1}\^{B\_{1}}a\_{i}\^{\\left(1\\right)}\\star k\_{ij}\^{\\left(1\\right)}+b\_{j}\^{\\left(1\\right)}\\in R\^{\\left(H\_{1}-h\_{1}+1\\right)\\times\\left(W\_{1}-w\_{1}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Azc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20%5Csum_%7Bi%3D1%7D%5E%7BB_%7B1%7D%7Da_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cstar%20k_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%2Bb_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28H_%7B1%7D-h_%7B1%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B1%7D-w_%7B1%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
zc_{j}^{\left(1\right)} & = & \sum_{i=1}^{B_{1}}a_{i}^{\left(1\right)}\star k_{ij}^{\left(1\right)}+b_{j}^{\left(1\right)}\in R^{\left(H_{1}-h_{1}+1\right)\times\left(W_{1}-w_{1}+1\right)}\end{aligned}")

![\\begin{aligned}
ac\_{j}\^{\\left(1\\right)} & = & f\\left(zc\_{j}\^{\\left(1\\right)}\\right)\\in R\^{\\left(H\_{1}-h\_{1}+1\\right)\\times\\left(W\_{1}-w\_{1}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20f%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H_%7B1%7D-h_%7B1%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B1%7D-w_%7B1%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
ac_{j}^{\left(1\right)} & = & f\left(zc_{j}^{\left(1\right)}\right)\in R^{\left(H_{1}-h_{1}+1\right)\times\left(W_{1}-w_{1}+1\right)}\end{aligned}")

![\\begin{aligned}
zp\_{j}\^{\\left(1\\right)} & = & maxpool\\left(ac\_{j}\^{\\left(1\\right)}\\right)\\in R\^{\\left(\\frac{H\_{1}-h\_{1}+1}{poolsize\_{1}}\\right)\\times\\left(\\frac{W\_{1}-w\_{1}+1}{poolsize\_{1}}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Azp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20maxpool%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH_%7B1%7D-h_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW_%7B1%7D-w_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
zp_{j}^{\left(1\right)} & = & maxpool\left(ac_{j}^{\left(1\right)}\right)\in R^{\left(\frac{H_{1}-h_{1}+1}{poolsize_{1}}\right)\times\left(\frac{W_{1}-w_{1}+1}{poolsize_{1}}\right)}\end{aligned}")

![\\begin{aligned}
ap\_{j}\^{\\left(1\\right)} & = & zp\_{j}\^{\\left(1\\right)}\\in R\^{\\left(\\frac{H\_{1}-h\_{1}+1}{poolsize\_{1}}\\right)\\times\\left(\\frac{W\_{1}-w\_{1}+1}{poolsize\_{1}}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%20%26%20%3D%20%26%20zp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH_%7B1%7D-h_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW_%7B1%7D-w_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
ap_{j}^{\left(1\right)} & = & zp_{j}^{\left(1\right)}\in R^{\left(\frac{H_{1}-h_{1}+1}{poolsize_{1}}\right)\times\left(\frac{W_{1}-w_{1}+1}{poolsize_{1}}\right)}\end{aligned}")

For I2-C2-MP2, we
have![H\_{2}=\\frac{H\_{1}-h\_{1}+1}{poolsize\_{1}}](https://latex.codecogs.com/png.latex?H_%7B2%7D%3D%5Cfrac%7BH_%7B1%7D-h_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D "H_{2}=\frac{H_{1}-h_{1}+1}{poolsize_{1}}"),
![W\_{2}=\\frac{W\_{1}-w\_{1}+1}{poolsize\_{1}}](https://latex.codecogs.com/png.latex?W_%7B2%7D%3D%5Cfrac%7BW_%7B1%7D-w_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D "W_{2}=\frac{W_{1}-w_{1}+1}{poolsize_{1}}"),
![B\_{2}=F\_{1}](https://latex.codecogs.com/png.latex?B_%7B2%7D%3DF_%7B1%7D "B_{2}=F_{1}"),
![j=1,\\cdots,F\_{2}](https://latex.codecogs.com/png.latex?j%3D1%2C%5Ccdots%2CF_%7B2%7D "j=1,\cdots,F_{2}")
where ![F\_{2}](https://latex.codecogs.com/png.latex?F_%7B2%7D "F_{2}")
is the number of convolution feature maps, and

![\\begin{aligned}
a\^{\\left(2\\right)} & = & ap\^{\\left(1\\right)}\\in R\^{H\_{2}\\times W\_{2}\\times B\_{2}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%282%5Cright%29%7D%20%26%20%3D%20%26%20ap%5E%7B%5Cleft%281%5Cright%29%7D%5Cin%20R%5E%7BH_%7B2%7D%5Ctimes%20W_%7B2%7D%5Ctimes%20B_%7B2%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(2\right)} & = & ap^{\left(1\right)}\in R^{H_{2}\times W_{2}\times B_{2}}\end{aligned}")

![\\begin{aligned}
zc\_{j}\^{\\left(2\\right)} & = & \\sum\_{i=1}\^{B\_{2}}a\_{i}\^{\\left(2\\right)}\\star k\_{ij}\^{\\left(2\\right)}+b\_{j}\^{\\left(2\\right)}\\in R\^{\\left(H\_{2}-h\_{2}+1\\right)\\times\\left(W\_{2}-w\_{2}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Azc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%20%26%20%3D%20%26%20%5Csum_%7Bi%3D1%7D%5E%7BB_%7B2%7D%7Da_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cstar%20k_%7Bij%7D%5E%7B%5Cleft%282%5Cright%29%7D%2Bb_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28H_%7B2%7D-h_%7B2%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B2%7D-w_%7B2%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
zc_{j}^{\left(2\right)} & = & \sum_{i=1}^{B_{2}}a_{i}^{\left(2\right)}\star k_{ij}^{\left(2\right)}+b_{j}^{\left(2\right)}\in R^{\left(H_{2}-h_{2}+1\right)\times\left(W_{2}-w_{2}+1\right)}\end{aligned}")

![\\begin{aligned}
ac\_{j}\^{\\left(2\\right)} & = & f\\left(zc\_{j}\^{\\left(2\\right)}\\right)\\in R\^{\\left(H\_{2}-h\_{2}+1\\right)\\times\\left(W\_{2}-w\_{2}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%20%26%20%3D%20%26%20f%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H_%7B2%7D-h_%7B2%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B2%7D-w_%7B2%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
ac_{j}^{\left(2\right)} & = & f\left(zc_{j}^{\left(2\right)}\right)\in R^{\left(H_{2}-h_{2}+1\right)\times\left(W_{2}-w_{2}+1\right)}\end{aligned}")

![\\begin{aligned}
zp\_{j}\^{\\left(2\\right)} & = & maxpool\\left(ac\_{j}\^{\\left(2\\right)}\\right)\\in R\^{\\left(\\frac{H\_{2}-h\_{2}+1}{poolsize\_{2}}\\right)\\times\\left(\\frac{W\_{2}-w\_{2}+1}{poolsize\_{2}}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Azp_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%20%26%20%3D%20%26%20maxpool%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH_%7B2%7D-h_%7B2%7D%2B1%7D%7Bpoolsize_%7B2%7D%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW_%7B2%7D-w_%7B2%7D%2B1%7D%7Bpoolsize_%7B2%7D%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
zp_{j}^{\left(2\right)} & = & maxpool\left(ac_{j}^{\left(2\right)}\right)\in R^{\left(\frac{H_{2}-h_{2}+1}{poolsize_{2}}\right)\times\left(\frac{W_{2}-w_{2}+1}{poolsize_{2}}\right)}\end{aligned}")

![\\begin{aligned}
ap\_{j}\^{\\left(2\\right)} & = & zp\_{j}\^{\\left(2\\right)}\\in R\^{\\left(\\frac{H\_{2}-h\_{2}+1}{poolsize\_{2}}\\right)\\times\\left(\\frac{W\_{2}-w\_{2}+1}{poolsize\_{2}}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aap_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%20%26%20%3D%20%26%20zp_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH_%7B2%7D-h_%7B2%7D%2B1%7D%7Bpoolsize_%7B2%7D%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW_%7B2%7D-w_%7B2%7D%2B1%7D%7Bpoolsize_%7B2%7D%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
ap_{j}^{\left(2\right)} & = & zp_{j}^{\left(2\right)}\in R^{\left(\frac{H_{2}-h_{2}+1}{poolsize_{2}}\right)\times\left(\frac{W_{2}-w_{2}+1}{poolsize_{2}}\right)}\end{aligned}")

For I3-FC1-O, we have
![n\_{3}=\\left(\\frac{H\_{2}-h\_{2}+1}{poolsize\_{2}}\\right)\\times\\left(\\frac{W\_{2}-w\_{2}+1}{poolsize\_{2}}\\right)\\times F\_{2}](https://latex.codecogs.com/png.latex?n_%7B3%7D%3D%5Cleft%28%5Cfrac%7BH_%7B2%7D-h_%7B2%7D%2B1%7D%7Bpoolsize_%7B2%7D%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW_%7B2%7D-w_%7B2%7D%2B1%7D%7Bpoolsize_%7B2%7D%7D%5Cright%29%5Ctimes%20F_%7B2%7D "n_{3}=\left(\frac{H_{2}-h_{2}+1}{poolsize_{2}}\right)\times\left(\frac{W_{2}-w_{2}+1}{poolsize_{2}}\right)\times F_{2}")

![\\begin{aligned}
a\^{\\left(3\\right)} & = & reshape\\left(ap\^{\\left(2\\right)}\\right)\\in R\^{n\_{3}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%283%5Cright%29%7D%20%26%20%3D%20%26%20reshape%5Cleft%28ap%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(3\right)} & = & reshape\left(ap^{\left(2\right)}\right)\in R^{n_{3}\times1}\end{aligned}")

![\\begin{aligned}
z\^{\\left(4\\right)} & = & W\^{\\left(3\\right)T}a\^{\\left(3\\right)}+b\^{\\left(3\\right)},\\quad W\^{\\left(3\\right)}\\in R\^{n\_{3}\\times n\_{4}},b\^{\\left(3\\right)}\\in R\^{n\_{4}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Az%5E%7B%5Cleft%284%5Cright%29%7D%20%26%20%3D%20%26%20W%5E%7B%5Cleft%283%5Cright%29T%7Da%5E%7B%5Cleft%283%5Cright%29%7D%2Bb%5E%7B%5Cleft%283%5Cright%29%7D%2C%5Cquad%20W%5E%7B%5Cleft%283%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes%20n_%7B4%7D%7D%2Cb%5E%7B%5Cleft%283%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
z^{\left(4\right)} & = & W^{\left(3\right)T}a^{\left(3\right)}+b^{\left(3\right)},\quad W^{\left(3\right)}\in R^{n_{3}\times n_{4}},b^{\left(3\right)}\in R^{n_{4}\times1}\end{aligned}")

For **Euclidean** loss

![\\begin{aligned}
a\^{\\left(4\\right)} & = & f\\left(z\^{\\left(4\\right)}\\right)\\in R\^{n\_{4}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa%5E%7B%5Cleft%284%5Cright%29%7D%20%26%20%3D%20%26%20f%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
a^{\left(4\right)} & = & f\left(z^{\left(4\right)}\right)\in R^{n_{4}\times1}\end{aligned}")

![\\begin{aligned}
L & = & \\frac{1}{2}\\left|\\left|y-a\^{\\left(4\\right)}\\right|\\right|\_{2}\^{2}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20%5Cfrac%7B1%7D%7B2%7D%5Cleft%7C%5Cleft%7Cy-a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%7C%5Cright%7C_%7B2%7D%5E%7B2%7D%5Cend%7Baligned%7D "\begin{aligned}
L & = & \frac{1}{2}\left|\left|y-a^{\left(4\right)}\right|\right|_{2}^{2}\end{aligned}")

For **cross-entropy** loss

![\\begin{aligned}
a\_{i}\^{\\left(4\\right)} & = & \\frac{\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)}{\\sum\_{k=1}\^{C}\\exp\\left(z\_{k}\^{\\left(4\\right)}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Aa_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
a_{i}^{\left(4\right)} & = & \frac{\exp\left(z_{i}^{\left(4\right)}\right)}{\sum_{k=1}^{C}\exp\left(z_{k}^{\left(4\right)}\right)}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial a\_{i}\^{\\left(4\\right)}}{\\partial z\_{i}\^{\\left(4\\right)}} & = & \\frac{\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)\\sum\_{k=1}\^{C}\\exp\\left(z\_{k}\^{\\left(4\\right)}\\right)-\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)}{\\left(\\sum\_{k=1}\^{C}\\exp\\left(z\_{k}\^{\\left(4\\right)}\\right)\\right)\^{2}}=a\_{i}\^{\\left(4\\right)}\\left(1-a\_{i}\^{\\left(4\\right)}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Csum_%7Bk%3D1%7D%5E%7BC%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29-%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%7B%5Cleft%28%5Csum_%7Bk%3D1%7D%5E%7BC%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cright%29%5E%7B2%7D%7D%3Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial a_{i}^{\left(4\right)}}{\partial z_{i}^{\left(4\right)}} & = & \frac{\exp\left(z_{i}^{\left(4\right)}\right)\sum_{k=1}^{C}\exp\left(z_{k}^{\left(4\right)}\right)-\exp\left(z_{i}^{\left(4\right)}\right)\exp\left(z_{i}^{\left(4\right)}\right)}{\left(\sum_{k=1}^{C}\exp\left(z_{k}^{\left(4\right)}\right)\right)^{2}}=a_{i}^{\left(4\right)}\left(1-a_{i}^{\left(4\right)}\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial a\_{j}\^{\\left(4\\right)}}{\\partial z\_{i}\^{\\left(4\\right)}} & = & \\frac{-\\exp\\left(z\_{j}\^{\\left(4\\right)}\\right)\\exp\\left(z\_{i}\^{\\left(4\\right)}\\right)}{\\left(\\sum\_{k=1}\^{C}\\exp\\left(z\_{k}\^{\\left(4\\right)}\\right)\\right)\^{2}}=-a\_{j}\^{\\left(4\\right)}a\_{i}\^{\\left(4\\right)},\\quad j\\neq i\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20a_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20%5Cfrac%7B-%5Cexp%5Cleft%28z_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cexp%5Cleft%28z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%7B%5Cleft%28%5Csum_%7Bk%3D1%7D%5E%7BC%7D%5Cexp%5Cleft%28z_%7Bk%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cright%29%5E%7B2%7D%7D%3D-a_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%2C%5Cquad%20j%5Cneq%20i%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial a_{j}^{\left(4\right)}}{\partial z_{i}^{\left(4\right)}} & = & \frac{-\exp\left(z_{j}^{\left(4\right)}\right)\exp\left(z_{i}^{\left(4\right)}\right)}{\left(\sum_{k=1}^{C}\exp\left(z_{k}^{\left(4\right)}\right)\right)^{2}}=-a_{j}^{\left(4\right)}a_{i}^{\left(4\right)},\quad j\neq i\end{aligned}")

![\\begin{aligned}
L & = & -\\sum\_{i=1}\^{C}y\_{i}\\log\\left(a\_{i}\^{\\left(4\\right)}\\right)=-\\sum\_{j\\neq i}y\_{j}\\log\\left(a\_{j}\^{\\left(4\\right)}\\right)-y\_{i}\\log\\left(a\_{i}\^{\\left(4\\right)}\\right)\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AL%20%26%20%3D%20%26%20-%5Csum_%7Bi%3D1%7D%5E%7BC%7Dy_%7Bi%7D%5Clog%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%3D-%5Csum_%7Bj%5Cneq%20i%7Dy_%7Bj%7D%5Clog%5Cleft%28a_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29-y_%7Bi%7D%5Clog%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%5Cend%7Baligned%7D "\begin{aligned}
L & = & -\sum_{i=1}^{C}y_{i}\log\left(a_{i}^{\left(4\right)}\right)=-\sum_{j\neq i}y_{j}\log\left(a_{j}^{\left(4\right)}\right)-y_{i}\log\left(a_{i}^{\left(4\right)}\right)\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial a\_{i}\^{\\left(4\\right)}} & = & -\\frac{y\_{i}}{a\_{i}\^{\\left(4\\right)}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20-%5Cfrac%7By_%7Bi%7D%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial a_{i}^{\left(4\right)}} & = & -\frac{y_{i}}{a_{i}^{\left(4\right)}}\end{aligned}")

![\\begin{aligned}
\\Rightarrow\\delta\^{\\left(a\^{\\left(4\\right)}\\right)} & = & -\\frac{y}{a\^{\\left(4\\right)}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5CRightarrow%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20-%5Cfrac%7By%7D%7Ba%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\Rightarrow\delta^{\left(a^{\left(4\right)}\right)} & = & -\frac{y}{a^{\left(4\right)}}\end{aligned}")

![\\begin{aligned}
\\frac{\\partial L}{\\partial z\_{i}\^{\\left(4\\right)}} & = & -\\sum\_{j\\neq i}y\_{j}\\frac{1}{a\_{j}\^{\\left(4\\right)}}\\left(-a\_{j}\^{\\left(4\\right)}a\_{i}\^{\\left(4\\right)}\\right)-y\_{i}\\frac{1}{a\_{i}\^{\\left(4\\right)}}a\_{i}\^{\\left(4\\right)}\\left(1-a\_{i}\^{\\left(4\\right)}\\right)=\\sum\_{j\\neq i}y\_{j}a\_{i}\^{\\left(4\\right)}-y\_{i}+y\_{i}a\_{i}\^{\\left(4\\right)}=a\_{i}\^{\\left(4\\right)}-y\_{i}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%20%26%20%3D%20%26%20-%5Csum_%7Bj%5Cneq%20i%7Dy_%7Bj%7D%5Cfrac%7B1%7D%7Ba_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cleft%28-a_%7Bj%7D%5E%7B%5Cleft%284%5Cright%29%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29-y_%7Bi%7D%5Cfrac%7B1%7D%7Ba_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cleft%281-a_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%3D%5Csum_%7Bj%5Cneq%20i%7Dy_%7Bj%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D-y_%7Bi%7D%2By_%7Bi%7Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D%3Da_%7Bi%7D%5E%7B%5Cleft%284%5Cright%29%7D-y_%7Bi%7D%5Cend%7Baligned%7D "\begin{aligned}
\frac{\partial L}{\partial z_{i}^{\left(4\right)}} & = & -\sum_{j\neq i}y_{j}\frac{1}{a_{j}^{\left(4\right)}}\left(-a_{j}^{\left(4\right)}a_{i}^{\left(4\right)}\right)-y_{i}\frac{1}{a_{i}^{\left(4\right)}}a_{i}^{\left(4\right)}\left(1-a_{i}^{\left(4\right)}\right)=\sum_{j\neq i}y_{j}a_{i}^{\left(4\right)}-y_{i}+y_{i}a_{i}^{\left(4\right)}=a_{i}^{\left(4\right)}-y_{i}\end{aligned}")

![\\begin{aligned}
\\Rightarrow & \\delta\^{\\left(z\^{\\left(4\\right)}\\right)}= & \\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}=a\^{\\left(4\\right)}-y\\in R\^{n\_{4}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5CRightarrow%20%26%20%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%284%5Cright%29%7D-y%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\Rightarrow & \delta^{\left(z^{\left(4\right)}\right)}= & \frac{\partial L}{\partial z^{\left(4\right)}}=a^{\left(4\right)}-y\in R^{n_{4}\times1}\end{aligned}")

![\\begin{aligned}
\\delta\^{\\left(a\^{\\left(3\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}\\frac{\\partial z\^{\\left(4\\right)}}{\\partial a\^{\\left(3\\right)}}=W\^{\\left(3\\right)}\\delta\^{\\left(z\^{\\left(4\\right)}\\right)}\\in R\^{n\_{3}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20a%5E%7B%5Cleft%283%5Cright%29%7D%7D%3DW%5E%7B%5Cleft%283%5Cright%29%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a^{\left(3\right)}\right)} & = & \frac{\partial L}{\partial a^{\left(3\right)}}=\frac{\partial L}{\partial z^{\left(4\right)}}\frac{\partial z^{\left(4\right)}}{\partial a^{\left(3\right)}}=W^{\left(3\right)}\delta^{\left(z^{\left(4\right)}\right)}\in R^{n_{3}\times1}\end{aligned}")

![\\begin{aligned}
\\nabla\_{W\^{\\left(3\\right)}}L & = & \\frac{\\partial L}{\\partial W\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}\\frac{\\partial z\^{\\left(4\\right)}}{\\partial W\^{\\left(3\\right)}}=a\^{\\left(3\\right)T}\\delta\^{\\left(z\^{\\left(4\\right)}\\right)}\\in R\^{n\_{3}\\times n\_{4}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7BW%5E%7B%5Cleft%283%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20W%5E%7B%5Cleft%283%5Cright%29%7D%7D%3Da%5E%7B%5Cleft%283%5Cright%29T%7D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes%20n_%7B4%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{W^{\left(3\right)}}L & = & \frac{\partial L}{\partial W^{\left(3\right)}}=\frac{\partial L}{\partial z^{\left(4\right)}}\frac{\partial z^{\left(4\right)}}{\partial W^{\left(3\right)}}=a^{\left(3\right)T}\delta^{\left(z^{\left(4\right)}\right)}\in R^{n_{3}\times n_{4}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{b\^{\\left(3\\right)}}L & = & \\frac{\\partial L}{\\partial b\^{\\left(3\\right)}}=\\frac{\\partial L}{\\partial z\^{\\left(4\\right)}}\\frac{\\partial z\^{\\left(4\\right)}}{\\partial b\^{\\left(3\\right)}}=\\delta\^{\\left(z\^{\\left(4\\right)}\\right)}\\in R\^{n\_{4}\\times1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bb%5E%7B%5Cleft%283%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7B%5Cleft%284%5Cright%29%7D%7D%7B%5Cpartial%20b%5E%7B%5Cleft%283%5Cright%29%7D%7D%3D%5Cdelta%5E%7B%5Cleft%28z%5E%7B%5Cleft%284%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B4%7D%5Ctimes1%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{b^{\left(3\right)}}L & = & \frac{\partial L}{\partial b^{\left(3\right)}}=\frac{\partial L}{\partial z^{\left(4\right)}}\frac{\partial z^{\left(4\right)}}{\partial b^{\left(3\right)}}=\delta^{\left(z^{\left(4\right)}\right)}\in R^{n_{4}\times1}\end{aligned}")

Here,
![\\delta\^{\\left(a\^{\\left(3\\right)}\\right)}\\in R\^{n\_{3}\\times1}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28a%5E%7B%5Cleft%283%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7Bn_%7B3%7D%5Ctimes1%7D "\delta^{\left(a^{\left(3\right)}\right)}\in R^{n_{3}\times1}")
is the error sensitivity of the reshaped output of the second maxpooling
layer. Thus we have get the
![\\delta\^{\\left(ap\_{j}\^{\\left(2\\right)}\\right)}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28ap_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D "\delta^{\left(ap_{j}^{\left(2\right)}\right)}"),
then we get

![\\begin{aligned}
\\delta\^{\\left(zp\_{j}\^{\\left(2\\right)}\\right)} & = & \\delta\^{\\left(ap\_{j}\^{\\left(2\\right)}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28zp_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cdelta%5E%7B%5Cleft%28ap_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(zp_{j}^{\left(2\right)}\right)} & = & \delta^{\left(ap_{j}^{\left(2\right)}\right)}\end{aligned}")

then, we upsample the error sensitivity and get

![\\begin{aligned}
\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)} & = & up\\left(\\delta\^{\\left(zp\_{j}\^{\\left(2\\right)}\\right)}\\right)\\in R\^{\\left(H\_{2}-k\_{2}+1\\right)\\times\\left(W\_{2}-w\_{2}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20up%5Cleft%28%5Cdelta%5E%7B%5Cleft%28zp_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H_%7B2%7D-k_%7B2%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B2%7D-w_%7B2%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(ac_{j}^{\left(1\right)}\right)} & = & up\left(\delta^{\left(zp_{j}^{\left(2\right)}\right)}\right)\in R^{\left(H_{2}-k_{2}+1\right)\times\left(W_{2}-w_{2}+1\right)}\end{aligned}")

then, we continue backpropagate the error sensitity

![\\begin{aligned}
\\delta\^{\\left(zc\_{j}\^{\\left(2\\right)}\\right)} & = & \\delta\^{\\left(ac\_{j}\^{\\left(2\\right)}\\right)}\\circ ac\_{j}\^{\\left(2\\right)}\\circ\\left(1-ac\_{j}\^{\\left(2\\right)}\\right)\\in R\^{\\left(H\_{2}-h\_{2}+1\\right)\\times\\left(W\_{2}-w\_{2}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Ccirc%20ac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Ccirc%5Cleft%281-ac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H_%7B2%7D-h_%7B2%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B2%7D-w_%7B2%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(zc_{j}^{\left(2\right)}\right)} & = & \delta^{\left(ac_{j}^{\left(2\right)}\right)}\circ ac_{j}^{\left(2\right)}\circ\left(1-ac_{j}^{\left(2\right)}\right)\in R^{\left(H_{2}-h_{2}+1\right)\times\left(W_{2}-w_{2}+1\right)}\end{aligned}")

then, we use the following operations to get the error sensity of
![a\_{i}\^{\\left(2\\right)}](https://latex.codecogs.com/png.latex?a_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D "a_{i}^{\left(2\right)}")

![\\begin{aligned}
\\delta\^{\\left(a\_{i}\^{\\left(2\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\_{i}\^{\\left(2\\right)}}=\\sum\_{j=1}\^{F\_{2}}\\frac{\\partial L}{\\partial zc\_{j}\^{\\left(2\\right)}}\\frac{\\partial zc\_{j}\^{\\left(2\\right)}}{\\partial a\_{i}\^{\\left(2\\right)}}=\\sum\_{j=1}\^{F\_{2}}conv2\\left(\\delta\^{\\left(zc\_{j}\^{\\left(2\\right)}\\right)},\\ rot180\\left(k\_{ij}\^{\\left(2\\right)}\\right),\\ 'full'\\right)\\in R\^{H\_{2}\\times W\_{2}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BF_%7B2%7D%7D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BF_%7B2%7D%7Dconv2%5Cleft%28%5Cdelta%5E%7B%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%2C%5C%20rot180%5Cleft%28k_%7Bij%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%2C%5C%20%27full%27%5Cright%29%5Cin%20R%5E%7BH_%7B2%7D%5Ctimes%20W_%7B2%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a_{i}^{\left(2\right)}\right)} & = & \frac{\partial L}{\partial a_{i}^{\left(2\right)}}=\sum_{j=1}^{F_{2}}\frac{\partial L}{\partial zc_{j}^{\left(2\right)}}\frac{\partial zc_{j}^{\left(2\right)}}{\partial a_{i}^{\left(2\right)}}=\sum_{j=1}^{F_{2}}conv2\left(\delta^{\left(zc_{j}^{\left(2\right)}\right)},\ rot180\left(k_{ij}^{\left(2\right)}\right),\ 'full'\right)\in R^{H_{2}\times W_{2}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{k\_{ij}\^{\\left(2\\right)}}L & = & \\frac{\\partial L}{\\partial k\_{ij}\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial zc\_{j}\^{\\left(2\\right)}}\\frac{\\partial zc\_{j}\^{\\left(2\\right)}}{\\partial k\_{ij}\^{\\left(2\\right)}}=rot180\\left(conv2\\left(a\_{i}\^{\\left(2\\right)},rot180\\left(\\delta\^{\\left(ac\_{j}\^{\\left(2\\right)}\\right)}\\right),'valid'\\right)\\right)\\in R\^{h\_{2}\\times w\_{2}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bk_%7Bij%7D%5E%7B%5Cleft%282%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20k_%7Bij%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20k_%7Bij%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%3Drot180%5Cleft%28conv2%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D%2Crot180%5Cleft%28%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cright%29%2C%27valid%27%5Cright%29%5Cright%29%5Cin%20R%5E%7Bh_%7B2%7D%5Ctimes%20w_%7B2%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{k_{ij}^{\left(2\right)}}L & = & \frac{\partial L}{\partial k_{ij}^{\left(2\right)}}=\frac{\partial L}{\partial zc_{j}^{\left(2\right)}}\frac{\partial zc_{j}^{\left(2\right)}}{\partial k_{ij}^{\left(2\right)}}=rot180\left(conv2\left(a_{i}^{\left(2\right)},rot180\left(\delta^{\left(ac_{j}^{\left(2\right)}\right)}\right),'valid'\right)\right)\in R^{h_{2}\times w_{2}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{b\_{j}\^{\\left(2\\right)}}L & = & \\frac{\\partial L}{\\partial b\_{j}\^{\\left(2\\right)}}=\\frac{\\partial L}{\\partial zc\_{j}\^{\\left(2\\right)}}\\frac{\\partial zc\_{j}\^{\\left(2\\right)}}{\\partial b\_{j}\^{\\left(2\\right)}}=\\sum\_{u,v}\\left(\\delta\^{\\left(ac\_{j}\^{\\left(2\\right)}\\right)}\\right)\_{u,v}\\in R\^{1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bb_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%7D%3D%5Csum_%7Bu%2Cv%7D%5Cleft%28%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cright%29_%7Bu%2Cv%7D%5Cin%20R%5E%7B1%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{b_{j}^{\left(2\right)}}L & = & \frac{\partial L}{\partial b_{j}^{\left(2\right)}}=\frac{\partial L}{\partial zc_{j}^{\left(2\right)}}\frac{\partial zc_{j}^{\left(2\right)}}{\partial b_{j}^{\left(2\right)}}=\sum_{u,v}\left(\delta^{\left(ac_{j}^{\left(2\right)}\right)}\right)_{u,v}\in R^{1}\end{aligned}")

From
![\\delta\^{\\left(a\_{i}\^{\\left(2\\right)}\\right)}\\in R\^{H\_{2}\\times W\_{2}}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%282%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7BH_%7B2%7D%5Ctimes%20W_%7B2%7D%7D "\delta^{\left(a_{i}^{\left(2\right)}\right)}\in R^{H_{2}\times W_{2}}"),
we could get
![\\delta\^{\\left(ap\_{j}\^{\\left(1\\right)}\\right)}\\in R\^{\\left(\\frac{H\_{1}-h\_{1}+1}{poolsize\_{1}}\\right)\\times\\left(\\frac{W\_{1}-w\_{1}+1}{poolsize\_{1}}\\right)}](https://latex.codecogs.com/png.latex?%5Cdelta%5E%7B%5Cleft%28ap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cin%20R%5E%7B%5Cleft%28%5Cfrac%7BH_%7B1%7D-h_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D%5Cright%29%5Ctimes%5Cleft%28%5Cfrac%7BW_%7B1%7D-w_%7B1%7D%2B1%7D%7Bpoolsize_%7B1%7D%7D%5Cright%29%7D "\delta^{\left(ap_{j}^{\left(1\right)}\right)}\in R^{\left(\frac{H_{1}-h_{1}+1}{poolsize_{1}}\right)\times\left(\frac{W_{1}-w_{1}+1}{poolsize_{1}}\right)}"),
then

![\\begin{aligned}
\\delta\^{\\left(zp\_{j}\^{\\left(1\\right)}\\right)} & = & \\delta\^{\\left(ap\_{j}\^{\\left(1\\right)}\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28zp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cdelta%5E%7B%5Cleft%28ap_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(zp_{j}^{\left(1\right)}\right)} & = & \delta^{\left(ap_{j}^{\left(1\right)}\right)}\end{aligned}")

then, we upsample the error sensitity and have

![\\begin{aligned}
\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)} & = & up\\left(\\delta\^{\\left(zp\_{j}\^{\\left(1\\right)}\\right)}\\right)\\in R\^{\\left(H\_{1}-h\_{1}+1\\right)\\times\\left(W\_{1}-w\_{1}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20up%5Cleft%28%5Cdelta%5E%7B%5Cleft%28zp_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H_%7B1%7D-h_%7B1%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B1%7D-w_%7B1%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(ac_{j}^{\left(1\right)}\right)} & = & up\left(\delta^{\left(zp_{j}^{\left(1\right)}\right)}\right)\in R^{\left(H_{1}-h_{1}+1\right)\times\left(W_{1}-w_{1}+1\right)}\end{aligned}")

then, we continue backpropagate the error sensitity

![\\begin{aligned}
\\delta\^{\\left(zc\_{j}\^{\\left(1\\right)}\\right)} & = & \\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)}\\circ ac\_{j}\^{\\left(1\\right)}\\circ\\left(1-ac\_{j}\^{\\left(1\\right)}\\right)\\in R\^{\\left(H\_{1}-h\_{1}+1\\right)\\times\\left(W\_{1}-w\_{1}+1\\right)}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Ccirc%20ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Ccirc%5Cleft%281-ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%5Cin%20R%5E%7B%5Cleft%28H_%7B1%7D-h_%7B1%7D%2B1%5Cright%29%5Ctimes%5Cleft%28W_%7B1%7D-w_%7B1%7D%2B1%5Cright%29%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(zc_{j}^{\left(1\right)}\right)} & = & \delta^{\left(ac_{j}^{\left(1\right)}\right)}\circ ac_{j}^{\left(1\right)}\circ\left(1-ac_{j}^{\left(1\right)}\right)\in R^{\left(H_{1}-h_{1}+1\right)\times\left(W_{1}-w_{1}+1\right)}\end{aligned}")

then, we use the following operations to get the error sensity of
![a\_{i}\^{\\left(1\\right)}](https://latex.codecogs.com/png.latex?a_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D "a_{i}^{\left(1\right)}")

![\\begin{aligned}
\\delta\^{\\left(a\_{i}\^{\\left(1\\right)}\\right)} & = & \\frac{\\partial L}{\\partial a\_{i}\^{\\left(1\\right)}}=\\sum\_{j=1}\^{F\_{1}}\\frac{\\partial L}{\\partial zc\_{j}\^{\\left(1\\right)}}\\frac{\\partial zc\_{j}\^{\\left(1\\right)}}{\\partial a\_{i}\^{\\left(1\\right)}}=\\sum\_{j=1}\^{F\_{1}}conv2\\left(\\delta\^{\\left(zc\_{j}\^{\\left(1\\right)}\\right)},\\ rot180\\left(k\_{ij}\^{\\left(1\\right)}\\right),\\ 'full'\\right)\\in R\^{H\_{1}\\times W\_{1}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cdelta%5E%7B%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BF_%7B1%7D%7D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%7B%5Cpartial%20a_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BF_%7B1%7D%7Dconv2%5Cleft%28%5Cdelta%5E%7B%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%2C%5C%20rot180%5Cleft%28k_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%2C%5C%20%27full%27%5Cright%29%5Cin%20R%5E%7BH_%7B1%7D%5Ctimes%20W_%7B1%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\delta^{\left(a_{i}^{\left(1\right)}\right)} & = & \frac{\partial L}{\partial a_{i}^{\left(1\right)}}=\sum_{j=1}^{F_{1}}\frac{\partial L}{\partial zc_{j}^{\left(1\right)}}\frac{\partial zc_{j}^{\left(1\right)}}{\partial a_{i}^{\left(1\right)}}=\sum_{j=1}^{F_{1}}conv2\left(\delta^{\left(zc_{j}^{\left(1\right)}\right)},\ rot180\left(k_{ij}^{\left(1\right)}\right),\ 'full'\right)\in R^{H_{1}\times W_{1}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{k\_{ij}\^{\\left(1\\right)}}L & = & \\frac{\\partial L}{\\partial k\_{ij}\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial zc\_{j}\^{\\left(1\\right)}}\\frac{\\partial zc\_{j}\^{\\left(1\\right)}}{\\partial k\_{ij}\^{\\left(1\\right)}}=rot180\\left(conv2\\left(a\_{i}\^{\\left(1\\right)},rot180\\left(\\delta\^{\\left(zc\_{j}\^{\\left(1\\right)}\\right)}\\right),'valid'\\right)\\right)\\in R\^{h\_{1}\\times w\_{1}}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bk_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20k_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%7B%5Cpartial%20k_%7Bij%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3Drot180%5Cleft%28conv2%5Cleft%28a_%7Bi%7D%5E%7B%5Cleft%281%5Cright%29%7D%2Crot180%5Cleft%28%5Cdelta%5E%7B%5Cleft%28zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cright%29%2C%27valid%27%5Cright%29%5Cright%29%5Cin%20R%5E%7Bh_%7B1%7D%5Ctimes%20w_%7B1%7D%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{k_{ij}^{\left(1\right)}}L & = & \frac{\partial L}{\partial k_{ij}^{\left(1\right)}}=\frac{\partial L}{\partial zc_{j}^{\left(1\right)}}\frac{\partial zc_{j}^{\left(1\right)}}{\partial k_{ij}^{\left(1\right)}}=rot180\left(conv2\left(a_{i}^{\left(1\right)},rot180\left(\delta^{\left(zc_{j}^{\left(1\right)}\right)}\right),'valid'\right)\right)\in R^{h_{1}\times w_{1}}\end{aligned}")

![\\begin{aligned}
\\nabla\_{b\_{j}\^{\\left(1\\right)}}L & = & \\frac{\\partial L}{\\partial b\_{j}\^{\\left(1\\right)}}=\\frac{\\partial L}{\\partial zc\_{j}\^{\\left(1\\right)}}\\frac{\\partial zc\_{j}\^{\\left(1\\right)}}{\\partial b\_{j}\^{\\left(1\\right)}}=\\sum\_{u,v}\\left(\\delta\^{\\left(ac\_{j}\^{\\left(1\\right)}\\right)}\\right)\_{u,v}\\in R\^{1}\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bb_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7DL%20%26%20%3D%20%26%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%5Cfrac%7B%5Cpartial%20zc_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%7B%5Cpartial%20b_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%7D%3D%5Csum_%7Bu%2Cv%7D%5Cleft%28%5Cdelta%5E%7B%5Cleft%28ac_%7Bj%7D%5E%7B%5Cleft%281%5Cright%29%7D%5Cright%29%7D%5Cright%29_%7Bu%2Cv%7D%5Cin%20R%5E%7B1%7D%5Cend%7Baligned%7D "\begin{aligned}
\nabla_{b_{j}^{\left(1\right)}}L & = & \frac{\partial L}{\partial b_{j}^{\left(1\right)}}=\frac{\partial L}{\partial zc_{j}^{\left(1\right)}}\frac{\partial zc_{j}^{\left(1\right)}}{\partial b_{j}^{\left(1\right)}}=\sum_{u,v}\left(\delta^{\left(ac_{j}^{\left(1\right)}\right)}\right)_{u,v}\in R^{1}\end{aligned}")


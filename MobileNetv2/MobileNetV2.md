[[mobilenetv2.pdf]]
#paper #vision #class 

Refer to [[Reading List]] for other papers

- Model based on DeepLabV3 Architecture
- Tries to achieve SOTA performance with simple network designs at the time of its release

# Preliminaries and Intuition

## **Depthwise Seperable Convolution** 
- It is a key aspect of this model
- Traditional convolution can be represented mathematically as:
	- An input tensor of dimensions $h_i × w_i × d_i$ (height x width x number of channels) on which we apply convolution using a kernel of size $R^{k \times k \times d_i \times d_j}$. This produces an output $h_i \times w_i \times d_j$.
	- This means they have a computational cost of $k \times k \times d_i \times d_j \times h_i \times w_i$.
	- We can reduce this with depthwise separable convolutions
- Depthwise separable convolutions work on the principle that all the filters in a convolution layer do not need to look at all the features in the input. Some features can only be dedicated to look at one section of the input, which will decrease the computational requirement but produce a similar, and in some cases a better result. 
	- This makes the filters only as deep as the depth section or group. 
	- Hence the cost becomes $h_i \times w_i \times d_i(k^2 + d_j)$ so
	- Explanation in this video:
	- https://www.youtube.com/watch?v=vVaRhZXovbw
	
## Linear Bottlenecks
- Linear bottlenecks are a method of reducing dimensionality while still being able to perform high dimensional computations by series of expansion and projections.
- They are amazing for fast small memory (cached systems like phones) since they do not have a large memory footprint
- They are based on the assumption that the layer activations lie on a manifold, and that manifold is representable in lower dimensional space. Hence we can reduce the dimensionality of a layer without affecting performance.
- It also tries to reduce the information loss by ReLU's linear transformative nature being applied on higher dimensional inputs. This takes advantage of the fact that since ReLU is a linear transformation, it will have lower information loss when applied on low-dimensional representations of high dimensional inputs.
- The ratio between the size of the input bottleneck and the expansion layer is called the expansion ratio (denoted by $t$).

## Inverted Residuals

	
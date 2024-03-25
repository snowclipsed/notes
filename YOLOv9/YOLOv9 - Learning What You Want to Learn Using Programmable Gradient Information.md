
# Abstract:
- States that most deep learning methods have problems with information bottlenecks.
	- Information bottleneck is when we compress high dimensional input data into lower dimension features by retaining the most important features to reduce computational complexity for training and inference, but at the same time it leads to data loss because we have to trim some features.
	- Although we try to minimize this information loss using regularization techniques the loss is still non-negligible.
- To deal with this memory loss, programmable gradient information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) are introduced
- Current SOTA

# Introduction
- Paper identifies 3 ways of dealing with data loss from information bottleneck.
	- **Reversibility** in deep learning is a method where we can compute/reconstruct the activations of a hidden/intermediate layer $Y_N$ from a layer $Y_{N+1}$ during backpropagation, hence we do not have to store the activations. This leads to more memory efficient networks.
	- **Masked modeling** paradigms may allow the model to extract features better and hence also learn better features, so they retain more information than usual even if they go through information bottlenecks. It utilizes reconstruction loss which measures how well is the model able to reconstruct the original sequence by predicting masked features.
	- **Deep supervision** is a method in deep learning which allows the hidden neurons to learn features more strongly by guiding the hidden layers directly by supplying them with an intermediate supervision signal aside the main supervision signal from backpropagation from the main target loss. This is implemented by providing auxiliary loss functions for intermediate layers. This is done by adding prediction layers between the intermediate hidden layers. It measures auxiliary loss between predictions from the intermediate layer themselves and ground truth. In the end these auxiliary functions are combined together with the main function with something like cross entropy loss.
		- Deep supervision facilitates gradient flow by mitigatiunderparameterisationng vanishing gradients, similar to residual networks. This allows for better representations learnt by the intermediate layers.
		- Allows for multi-task and more generalized learning.
		- Allows for more stable training
		- U-net architecture uses this.
- However all three paradigms have problems that need to be addressed:
	- Reversible architectures need additional layers hence the complexity of reversible architectures is higher to trade off with memory efficiency. Hence they require higher compute.
	- Masked Modeling's reconstruction loss may conflict with target loss. This causes masked modeling to produce incorrect input-target mappings.
	- Deep supervision leads to error accumulation. Error accumulation is when all the error values from the intermediate/auxiliary loss functions are accumulated with the target loss. In addition, if shallower layers are subject to information loss, the intermediate to output layers have naturally no method to retrieve it.
- To deal with all this, authors introduce PGI.
	- PGI generates more robust and reliable gradients through the reversible functions to learn deep features. This avoids semantic loss that occurs in normal deep supervision.
	- PGI architecture addition is built on the auxiliary branch (for auxiliary loss). Hence there is no added cost.
	- PGI can select loss functions on its own depending on the task.

Related work
- Yolov9 is SOTA currently
- Authors found high performance in many models with reversible architectures.
	- DynamicNet uses YoloV7 and merges it with CBNet architecture which has multi-level reversible branches with high parameter utilization.
	- YoloV9 builds on DynamicNet architecture to design reversible branches on which it implements PGI.
- Deep supervision works by two methods:
	- Guiding intermediate layers by introducing auxiliary losses for each intermediate layer using prediction layers between hidden layers.
	- Guiding feature maps to directly have properties that are present in the target image using depth or segmentation loss.
	- Deep supervision is usually not fit for lightweight models because it can cause underparameterization in them.
		- Underparameterization is a phenomenon where a model does not have enough learnable parameters relative to the complexity of the model it is trying to solve. If during deep supervision the given layer in the layer hierarchy does not have enough learnable parameters to calculate auxiliary loss, it can degrade the performance of that layer, and then this degraded output is fed to the next layer which may suffer from the same problem. Hence a cascading effect starts. This causes negative performance than original.
	- However PGI can reprogram semantic information and allows lightweight models to benefit from this (how??)

# Problem statement
- Says that vanishing gradient and gradient saturation is solved already (true), assumingly by residual layers and various normalization techniques.
- However deep nets still suffer from poor convergence and slow convergence.
- Through experiments they realize that this is because of information bottlenecks leading to information loss.

## Information Bottleneck Principle
- The information loss due to the bottleneck can be described by in terms of mutual information: $I(X,X) \ge I(X, f\theta(X) \ge I(X, g\phi(f\theta(X)))$ 
	- Here $I$ is the mutual information.
	- Mathematically, MI(X;Y) is defined  as:
	- $MI(X;Y) = ∑p(x,y) log2(p(x,y)/p(x)p(y))$
	- where $p(x)$ and $p(y)$ are the marginal probability distributions of X and $Y$, respectively, and $p(x,y)$ is their joint probability distribution. Intuitively, MI measures how much knowing the value of X reduces uncertainty about Y, or conversely, how much knowing the value of Y reduces uncertainty about X.
	- $f$ and $g$ are transformation functions with trainable parameters $\theta$ and $\phi$. 
- This represents that as more neural transformations are applied, the more information is lost. As in, deeper layers mean more information loss since there's consecutive application of transformation functions (neurons).
- This means a model with deeper layers retain lesser info about both the input and target. Hence it would naturally perform worse.
- A model with larger number of parameters has much more parameters and can learn larger number of features (information) about the data. This is why width is important in deep networks than depth itself.
- This increase in width can only increase the scope of learning more information by simply increasing the number of params, but the information loss per param is still the same (or, often increased because of more connections). 
- Authors propose to solve this problem.

## Reversible Functions
- If $r_\psi(X)$ is a function, it may have an inverse transformation $v_\zeta()$ which means when we apply $v_\zeta()$ on $r_\psi(X)$, we get $X$ back:
	- $X = v_\zeta(r_\psi(X))$, where $\psi$ and $\zeta$ are parameters
- A reversible function results in a perfect recreation of the initial data $X$, this means it has no information loss:
	- $I(X,X) = I(X, r_\psi(X)) = I(X, v_\zeta(r_\psi(X))$
-  Hence the activations can be recomputed through reversible functions. This leads to better performance. This can be mathematically represented as :
	- $X^{l+1} = X^l + f\theta ^{l+1}(X^l)$
	- In a PreAct ResNet model, this depicts the $l$ th layer and a transformation function $f$ is applied on the $l$-th layer. We can see that it is a reversible function as $X^{l+1}$ can be obtained by explicitly passing $X^l$ (Data from l-th layer) to the subsequent layers.
	- This leads to good convergence but high complexity. Hence why PreAct ResNet must need high amount of layers to function well (underparameterization).
- Authors experiment on masked modeling (transformers) and use approximation functions to approximate applying an inverse transformation on a masked modeling scenario.
	- $X = v_\zeta(r\psi(X)\cdot M)$
	- $M$ is a dynamic mask over the data $X$.
- We can pose the information bottleneck equation above as a mapping from input data $X$ to target $Y$:
	- $I(X,Y) \ge I(Y,X) \ge I(Y, f\theta(X) \ge \dots \ge I(Y, \hat Y)$ 
	- Because of underparameterization in the shallow layers, a lot of information can be lost in the first few layers itself during I(Y,X). Since if we lose information in the start, the succeeding transformation functions will have no way to recover the lost information.
	- Hence the goal for getting reliable gradients is minimizing information loss while mapping X to Y as in $I(Y,X)$ from $I(X,X)$

# Methodology
## Programmable Gradient Information
- Components of PGI:
	- Main branch
	- Auxiliary Reversible Branch
	- Multi-Level Auxiliary Information
- ![[Pasted image 20240229201721.png]]
- PGI doesn't add to inference cost because the auxiliary reversible branch (the grey thing) is not used in inference at all. 
- However, deep supervision based tasks are prone to error accumulation. To deal with this, authors introduce multi-level auxiliary information.
### Auxiliary Reversible Branch
- Idea is to guide the hidden layers to learn features more relevant to the target and avoid false correlations. Taken from deep supervision.
- However, they are reversible functions, hence they do not have information loss by applying transformation functions. 
- To do this in usual deep supervision, they need to add additional neurons for loss prediction between the hidden layers. This leads to increased inference costs (by minimum 20% from normal).
- However, we can avoid this by simply excluding the auxiliary branch from inference. 
- This method also combats underparameterisation in shallower networks since it doesn't pass the entire original information again, but simply adds new information. Authors call this auxiliary supervision. Hence PGI can be applied to shallower networks.
### Multi-Level Auxiliary Information
- Feature pyramids are used in object detection for multi-task detection.
- Deep supervision allows shallower layers to learn features corresponding to small objects, and treats other objects as the background. This causes information loss for the deeper layers.
- Authors believe all layers should receive information about all targets to prevent information loss through maintaining information completeness.
- Multi-level aux information is an integration network inserted between the pyramid.
	- An Integration Network is a type of ANN architecture that is designed to learn a hierarchical representation of data by combining information from multiple levels of abstraction in a recursive manner. These networks integrate information across different levels or scales of representation. E.g. small and large image features in object detection.
- It combines gradients from prediction heads in the integration network, and then passes it back to the network.
- This allows for the main gradient to not be influenced by a single object.

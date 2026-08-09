# FGSM on MNIST: the maths, step by step

This is a companion to [fgsm.ipynb](./fgsm.ipynb). It explains how the MNIST
classifier is trained, what gradient the attack calculates, why taking the sign
of that gradient makes sense, and how the equations become PyTorch operations.

The implementation follows the
[PyTorch FGSM tutorial](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html),
the [PyTorch MNIST example](https://github.com/pytorch/examples/tree/main/mnist),
and the original
[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
paper.

The central idea is:

> Normal training changes the model parameters to reduce classification loss.
> FGSM holds those parameters fixed and changes the input pixels in the
> direction that increases the loss.

## 1. The complete experiment

The notebook performs two separate jobs.

### Job A: train a classifier

~~~text
MNIST image
    ↓
CNN
    ↓
10 class scores
    ↓
classification loss
    ↓
update model parameters
~~~

### Job B: attack the trained classifier

~~~text
MNIST image
    ↓
fixed trained CNN
    ↓
classification loss
    ↓
gradient with respect to the image
    ↓
slightly change every pixel
    ↓
classify the changed image
~~~

The distinction between these jobs is the most important thing to understand:

| Phase | What is changed? | What is held fixed? | Goal |
|---|---|---|---|
| Training | Model parameters | Training images | Reduce loss |
| FGSM attack | Input pixels | Model parameters | Increase loss |

## 2. Notation

We use the following symbols:

| Symbol | Read as | Meaning |
|---|---|---|
| $x$ | “x” | One original MNIST image |
| $y$ | “y” | The image's correct digit label |
| $\theta$ | “theta” | All model weights and biases |
| $f_\theta(x)$ | “f theta of x” | Model output for image $x$ |
| $J(\theta,x,y)$ | “J of theta, x, and y” | Classification loss |
| $\nabla_\theta J$ | “gradient of J with respect to theta” | How parameter changes affect loss |
| $\nabla_x J$ | “gradient of J with respect to x” | How pixel changes affect loss |
| $\epsilon$ | “epsilon” | Maximum FGSM pixel change |
| $x_{\text{adv}}$ | “x adversarial” | The perturbed image |

The subscript on a gradient tells us what we are allowed to change.

$$
\nabla_\theta J
\qquad\text{versus}\qquad
\nabla_x J
$$

Read this as: “The gradient of the loss with respect to the parameters, versus
the gradient of the loss with respect to the input.”

They come from the same loss and the same backward-pass machinery, but they
answer different questions.

## 3. The MNIST input

Each MNIST image contains one grayscale digit and has shape:

$$
1\times28\times28
$$

Read this as: “One channel, twenty-eight pixels high, and twenty-eight pixels
wide.”

One image therefore contains:

$$
28\times28=784
$$

pixel values. Before normalization, each pixel lies in:

$$
x_i\in[0,1]
$$

Read this as: “Pixel $i$ is between zero and one.” Zero is black and one is
white.

A training batch has shape:

$$
[B,1,28,28]
$$

where $B$ is the batch size. With <code>BATCH_SIZE = 64</code>, the shape is
<code>[64, 1, 28, 28]</code>.

### Labels

Each target is an integer:

$$
y\in\{0,1,2,\ldots,9\}
$$

Read this as: “The true label is one of the ten digits from zero through nine.”

## 4. Normalization

The model is trained on normalized pixels rather than the original pixel values.
For one pixel:

$$
z=\frac{x-\mu}{\sigma}
$$

Read this as:

> “The normalized value $z$ equals the original pixel $x$ minus the mean
> $\mu$, divided by the standard deviation $\sigma$.”

The notebook uses:

$$
\mu=0.1307
\qquad\text{and}\qquad
\sigma=0.3081
$$

so:

$$
z=\frac{x-0.1307}{0.3081}
$$

The inverse operation is:

$$
x=z\sigma+\mu
$$

Read this as: “Recover the original pixel by multiplying the normalized value
by the standard deviation and adding the mean.”

These equations correspond to:

~~~python
def normalize_pixels(images):
    return (images - MNIST_MEAN) / MNIST_STD


def denormalize(images):
    return images * MNIST_STD + MNIST_MEAN
~~~

### Why the attack returns to the original pixel scale

The notebook defines epsilon in ordinary pixel units. For example,
$\epsilon=0.1$ permits each pixel to move by at most one tenth of the complete
black-to-white range.

The attack therefore:

1. reverses normalization;
2. calculates the attack in the original $[0,1]$ pixel range;
3. clips the changed pixels to $[0,1]$;
4. normalizes the result before passing it through the classifier.

This keeps epsilon understandable and makes clipping mathematically correct.

## 5. How the classifier transforms an image

For a batch size $B$, the network changes tensor shapes as follows:

| Operation | Output shape | What it does |
|---|---:|---|
| Input | $[B,1,28,28]$ | Grayscale images |
| <code>conv1</code> | $[B,32,26,26]$ | Finds 32 local feature types |
| <code>conv2</code> | $[B,64,24,24]$ | Builds 64 richer feature maps |
| Max pool | $[B,64,12,12]$ | Halves height and width |
| Flatten | $[B,9216]$ | Converts feature maps to vectors |
| <code>fc1</code> | $[B,128]$ | Combines image features |
| <code>fc2</code> | $[B,10]$ | Produces one score per digit |
| Log softmax | $[B,10]$ | Produces log-probabilities |

The flattened size is:

$$
64\times12\times12=9216
$$

which explains:

~~~python
self.fc1 = nn.Linear(9_216, 128)
~~~

### A convolution

At a simplified level, one convolution output is:

$$
h_{c,i,j}
=
b_c
+
\sum_{k,u,v}
w_{c,k,u,v}\,x_{k,i+u,j+v}
$$

Read this as:

> “Feature value $h$ for output channel $c$ and location $(i,j)$ equals a bias,
> plus the sum of nearby input pixels multiplied by learned filter weights.”

The indices mean:

- $c$: output feature channel;
- $k$: input channel;
- $i,j$: output image location;
- $u,v$: position inside the convolutional kernel.

The network learns filters whose responses are useful for identifying digit
parts such as edges, curves, and intersections.

### ReLU

After each convolution and the first fully connected layer, the notebook uses:

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

Read this as: “ReLU of $z$ is whichever is larger: zero or $z$.”

Negative activations become zero; positive activations pass through unchanged.

### Dropout

During training, dropout randomly sets some activations to zero. This discourages
the model from depending too heavily on particular features.

During evaluation and attack generation, the notebook calls:

~~~python
model.eval()
~~~

This disables dropout randomness. The same adversarial input should be evaluated
against a stable, deterministic model.

## 6. From class scores to probabilities

Let the final layer produce ten raw scores, or logits:

$$
q=
\begin{bmatrix}
q_0&q_1&\cdots&q_9
\end{bmatrix}
$$

The log-softmax for class $k$ is:

$$
\log p_k
=
q_k-\log\left(\sum_{j=0}^{9}e^{q_j}\right)
$$

Read this as:

> “The log-probability of class $k$ equals its score minus the logarithm of the
> sum of the exponentials of all ten scores.”

Softmax turns arbitrary scores into probabilities that are positive and sum to
one. Log-softmax returns their logarithms and is numerically convenient when
paired with negative log-likelihood loss.

The predicted digit is:

$$
\hat y=\operatorname*{arg\,max}_{k}q_k
$$

Read this as: “Y-hat is the class index $k$ whose score is largest.”

A hat usually means “an estimate,” so $\hat y$ is the model's estimated label.

In the notebook:

~~~python
prediction = output.argmax(dim=1)
~~~

## 7. Training loss

For one image with correct label $y$, negative log-likelihood loss is:

$$
J(\theta,x,y)=-\log p_\theta(y\mid x)
$$

Read this as:

> “The loss equals negative log of the probability that the model with
> parameters theta assigns to the correct label $y$, given image $x$.”

If the model assigns high probability to the correct digit, the loss is small.
If it assigns low probability to the correct digit, the loss is large.

For a batch of $B$ images:

$$
J_B(\theta)
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_\theta(y_i\mid x_i)
$$

Read this as:

> “The batch loss is negative one over the batch size times the sum of the
> correct-label log-probabilities.”

This is implemented by:

~~~python
output = model(data)
loss = F.nll_loss(output, target)
~~~

## 8. How normal training changes the model

During training, the images are treated as fixed examples and the model
parameters are changed.

The relevant gradient is:

$$
\nabla_\theta J(\theta,x,y)
$$

Read this as: “The gradient of the loss with respect to theta.”

Conceptually, a gradient-descent update is:

$$
\theta
\leftarrow
\theta-\alpha\nabla_\theta J
$$

Read this as:

> “Replace theta with its current value minus the learning rate alpha times the
> gradient of the loss with respect to theta.”

The gradient points toward increasing loss, so subtracting it moves toward
lower loss.

The notebook performs:

~~~python
optimizer.zero_grad(set_to_none=True)
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()
~~~

These lines mean:

1. clear old gradients;
2. calculate predictions and loss;
3. backpropagate to compute parameter gradients;
4. ask Adadelta to update the parameters.

Adadelta adaptively scales each parameter's update, so its exact update is more
involved than plain gradient descent. The gradient-descent equation still gives
the correct high-level mental model.

### Learning-rate schedule

After each epoch, the scheduler performs:

$$
\alpha_{e+1}=0.7\alpha_e
$$

Read this as: “The next epoch's learning rate equals seventy percent of the
current epoch's learning rate.”

The run begins at $1.0$, then uses $0.7$, $0.49$, $0.343$, and so on. Large
early steps learn quickly; smaller later steps refine the solution.

## 9. What was saved

After 14 epochs, the notebook saves:

~~~python
torch.save(model.state_dict(), model_path)
~~~

The file is:

~~~text
torch/models/mnist_cnn_state_dict.pt
~~~

A state dictionary maps parameter names to tensors:

~~~text
conv1.weight -> tensor
conv1.bias   -> tensor
...
fc2.weight   -> tensor
fc2.bias     -> tensor
~~~

It stores the learned numerical parameters, not the Python class definition.
Loading therefore requires creating the same architecture first:

~~~python
reloaded_model = Net().to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
reloaded_model.load_state_dict(state_dict)
reloaded_model.eval()
~~~

The notebook evaluates this fresh instance to prove that the saved artifact is
complete and usable. The executed run obtained 99.21% clean test accuracy before
and after reloading.

## 10. The attacker's goal

The notebook implements an **untargeted, white-box attack**.

- **Untargeted** means the attacker wants the prediction to become wrong but
  does not demand a particular wrong digit.
- **White-box** means the attacker knows the model and can calculate gradients
  through it.

The attacker wants to find a small perturbation $\delta$ that makes the loss
large:

$$
\underset{\delta}{\operatorname{maximize}}
\quad
J(\theta,x+\delta,y)
$$

subject to:

$$
\|\delta\|_\infty\leq\epsilon
$$

Read this as:

> “Choose perturbation delta to maximize the loss, while requiring delta's
> infinity norm to be no greater than epsilon.”

The symbol $\delta$ is read “delta” and means the change added to the image:

$$
x_{\text{adv}}=x+\delta
$$

### The infinity norm

For a vector of pixel changes:

$$
\|\delta\|_\infty=\max_i|\delta_i|
$$

Read this as: “The infinity norm of delta is the largest absolute change among
all pixels.”

Therefore:

$$
\|\delta\|_\infty\leq\epsilon
$$

means every individual pixel must satisfy:

$$
|\delta_i|\leq\epsilon
$$

This is a per-pixel maximum. It does not mean the average change is epsilon, and
it does not limit the sum of all pixel changes to epsilon.

## 11. Why the gradient points toward higher loss

The input gradient is:

$$
g=\nabla_xJ(\theta,x,y)
$$

Read this as: “Gradient $g$ equals the gradient of the loss with respect to the
input image.”

It has the same shape as the input:

$$
g\in\mathbb{R}^{1\times28\times28}
$$

Each element answers:

> “If this pixel increases by a tiny amount, how will the loss change?”

For pixel $i$:

$$
g_i=\frac{\partial J}{\partial x_i}
$$

Read this as: “Gradient component $i$ is the partial derivative of the loss
with respect to pixel $i$.”

- If $g_i>0$, increasing pixel $i$ tends to increase loss.
- If $g_i<0$, decreasing pixel $i$ tends to increase loss.
- If $g_i=0$, a tiny change in that pixel has no first-order effect.

## 12. Deriving FGSM

For a small input change $\delta$, a first-order Taylor approximation gives:

$$
J(\theta,x+\delta,y)
\approx
J(\theta,x,y)
+
\nabla_xJ(\theta,x,y)^\mathsf{T}\delta
$$

Read this as:

> “The loss at the changed image is approximately the original loss plus the
> input gradient transposed times the pixel change.”

The superscript $\mathsf{T}$ is read “transpose.” The product is a dot product:

$$
\nabla_xJ^\mathsf{T}\delta
=
\sum_i g_i\delta_i
$$

Read this as: “Sum, over all pixels, the input gradient component times that
pixel's change.”

The original loss $J(\theta,x,y)$ is fixed while choosing $\delta$. To increase
the approximate loss as much as possible, we maximize:

$$
\sum_i g_i\delta_i
$$

with each $\delta_i$ restricted to $[-\epsilon,\epsilon]$.

Consider one pixel:

$$
\underset{|\delta_i|\leq\epsilon}{\operatorname{maximize}}
\quad
g_i\delta_i
$$

There are two important cases:

- If $g_i$ is positive, choose $\delta_i=+\epsilon$.
- If $g_i$ is negative, choose $\delta_i=-\epsilon$.

Both cases are represented by:

$$
\delta_i=\epsilon\,\operatorname{sign}(g_i)
$$

Read this as: “Delta for pixel $i$ equals epsilon times the sign of gradient
component $i$.”

For the complete image:

$$
\delta
=
\epsilon\,
\operatorname{sign}
\left(
\nabla_xJ(\theta,x,y)
\right)
$$

and therefore:

$$
x_{\text{adv}}
=
x
+
\epsilon\,
\operatorname{sign}
\left(
\nabla_xJ(\theta,x,y)
\right)
$$

Read this as:

> “The adversarial image equals the original image plus epsilon times the sign
> of the gradient of the loss with respect to the image.”

This is the Fast Gradient Sign Method:

- **Fast**: it uses one backward pass and one perturbation step.
- **Gradient**: the input gradient identifies loss-increasing directions.
- **Sign**: every nonzero gradient component becomes either $-1$ or $+1$.

## 13. Why use the sign instead of the gradient magnitude?

Suppose three gradient components are:

$$
g=
\begin{bmatrix}
0.001&-4.2&0.3
\end{bmatrix}
$$

Their signs are:

$$
\operatorname{sign}(g)
=
\begin{bmatrix}
+1&-1&+1
\end{bmatrix}
$$

With $\epsilon=0.1$:

$$
\delta
=
0.1\operatorname{sign}(g)
=
\begin{bmatrix}
0.1&-0.1&0.1
\end{bmatrix}
$$

Every component uses the full permitted change in whichever direction increases
the first-order loss. This is the maximizing choice under the infinity-norm
constraint.

Using the raw gradient instead would produce changes with varying sizes and
would not automatically use the available per-pixel budget optimally.

## 14. A one-pixel numerical example

Suppose:

$$
x_i=0.40,\qquad
\epsilon=0.10,\qquad
\frac{\partial J}{\partial x_i}=-0.03
$$

The gradient sign is:

$$
\operatorname{sign}(-0.03)=-1
$$

The adversarial pixel is:

$$
x_{\text{adv},i}
=
0.40+0.10(-1)
=
0.30
$$

Read this as: “The negative gradient tells FGSM to reduce this pixel by the
maximum allowed amount.”

Now suppose another pixel is already close to white:

$$
x_j=0.97,\qquad
\operatorname{sign}(g_j)=+1
$$

Before clipping:

$$
0.97+0.10=1.07
$$

But valid pixels cannot exceed one, so:

$$
x_{\text{adv},j}
=
\operatorname{clip}_{[0,1]}(1.07)
=
1.00
$$

## 15. Clipping

The complete attack used by the notebook is:

$$
x_{\text{adv}}
=
\operatorname{clip}_{[0,1]}
\left(
x+\epsilon\operatorname{sign}(\nabla_xJ)
\right)
$$

Read this as:

> “Add epsilon times the input-gradient sign to the image, then clip every
> result to the valid pixel interval from zero to one.”

For one value:

$$
\operatorname{clip}_{[0,1]}(z)
=
\min(1,\max(0,z))
$$

Read this as: “First ensure $z$ is at least zero, then ensure it is at most one.”

The notebook implementation is:

~~~python
def fgsm_attack(images, epsilon, input_gradients):
    perturbed_images = images + epsilon * input_gradients.sign()
    return perturbed_images.clamp(0, 1)
~~~

## 16. How the notebook obtains input gradients

The core of <code>evaluate_fgsm()</code> can be understood as seven steps.

### Step 1: recover original pixels

~~~python
pixels = denormalize(normalized_data).clamp(0, 1).detach()
~~~

The data loader returns normalized tensors. This converts them back to the
original pixel range.

<code>detach()</code> starts a fresh computation graph. The attack does not need
the data-loading and denormalization history.

### Step 2: ask PyTorch to track pixel gradients

~~~python
pixels.requires_grad_(True)
~~~

Ordinary input tensors do not need gradients during model training, so PyTorch
does not retain them by default. This line says that pixels are now variables
whose effect on the loss must be measured.

### Step 3: normalize and classify

~~~python
output = model(normalize_pixels(pixels))
~~~

Normalization is inside the tracked computation. Backpropagation therefore
calculates the gradient all the way back to the original $[0,1]$ pixels.

### Step 4: calculate the true-label loss

~~~python
loss = F.nll_loss(output, target, reduction='sum')
~~~

FGSM increases the loss for the true labels. With batch-independent network
operations, summing losses still gives each image its own input gradient.

### Step 5: backpropagate to the pixels

~~~python
model.zero_grad(set_to_none=True)
loss.backward()
input_gradients = pixels.grad
~~~

After <code>backward()</code>, <code>pixels.grad</code> contains:

$$
\nabla_xJ(\theta,x,y)
$$

The model's parameters may also receive gradients during this calculation, but
there is no optimizer step. The parameters are therefore not changed.

### Step 6: create the adversarial image

~~~python
adversarial_pixels = fgsm_attack(
    pixels.detach(),
    epsilon,
    pixels.grad.detach(),
)
~~~

Both tensors are detached because attack construction does not need a
higher-order gradient graph.

### Step 7: classify the adversarial image

~~~python
with torch.no_grad():
    attacked_output = model(normalize_pixels(adversarial_pixels))
    final_prediction = attacked_output.argmax(dim=1)
~~~

The perturbed pixels are normalized because the model was trained on normalized
inputs. No gradient is needed for this final measurement.

## 17. Tensor shapes during the attack

With <code>ATTACK_BATCH_SIZE = 256</code>:

| Tensor | Shape | Meaning |
|---|---:|---|
| <code>normalized_data</code> | $[256,1,28,28]$ | Normalized clean images |
| <code>pixels</code> | $[256,1,28,28]$ | Clean images in $[0,1]$ |
| <code>output</code> | $[256,10]$ | Clean log-probabilities |
| <code>target</code> | $[256]$ | Correct labels |
| <code>pixels.grad</code> | $[256,1,28,28]$ | Input gradients |
| <code>adversarial_pixels</code> | $[256,1,28,28]$ | Perturbed images |
| <code>attacked_output</code> | $[256,10]$ | Adversarial log-probabilities |

The gradient must have the same shape as the input because it supplies one
partial derivative for every input value.

## 18. Parameter gradients versus input gradients

This comparison is worth memorizing:

| Question | Training | FGSM |
|---|---|---|
| Differentiated variable | $\theta$ | $x$ |
| Gradient | $\nabla_\theta J$ | $\nabla_x J$ |
| Direction used | Negative gradient | Positive gradient sign |
| Why? | Reduce loss | Increase loss |
| Update size | Chosen by Adadelta | Fixed per-pixel budget $\epsilon$ |
| Calls optimizer step? | Yes | No |

The two conceptual updates are:

$$
\text{training:}\qquad
\theta\leftarrow\theta-\alpha\nabla_\theta J
$$

$$
\text{attack:}\qquad
x_{\text{adv}}\leftarrow
x+\epsilon\operatorname{sign}(\nabla_xJ)
$$

Read these together as:

> “Training moves parameters against the parameter gradient to lower loss.
> FGSM moves pixels with the input-gradient sign to raise loss.”

## 19. Why tiny changes across many pixels can matter

MNIST has 784 pixels. Even when every individual change is small, many
loss-increasing changes can add together.

Using the Taylor approximation:

$$
\Delta J
\approx
\nabla_xJ^\mathsf{T}\delta
$$

and FGSM's choice:

$$
\delta_i=\epsilon\operatorname{sign}(g_i)
$$

gives:

$$
\Delta J
\approx
\epsilon\sum_i|g_i|
$$

Read this as:

> “The approximate loss increase equals epsilon times the sum of the absolute
> input-gradient components.”

To see why, each product becomes:

$$
g_i\delta_i
=
g_i\epsilon\operatorname{sign}(g_i)
=
\epsilon|g_i|
$$

Every nonzero pixel contribution is non-negative. Small aligned changes over
hundreds of dimensions can therefore produce a substantial change in the
network's internal activations.

This does not mean the attack is always visually invisible. Visibility depends
on epsilon, the image, display scaling, and human perception. Larger epsilon
values generally make perturbations easier to notice.

## 20. Measuring robust accuracy

Let:

$$
\hat y(x)=\operatorname*{arg\,max}_k f_\theta(x)_k
$$

be the clean prediction, and let $\hat y(x_{\text{adv}})$ be the adversarial
prediction.

The notebook counts an example as robust when it was initially correct and
remains correct:

$$
\mathbf{1}
\left[
\hat y(x)=y
\;\land\;
\hat y(x_{\text{adv}})=y
\right]
$$

Read this as:

> “The indicator is one when the clean prediction equals the true label and the
> adversarial prediction also equals the true label.”

The symbol $\land$ is read “and.” The indicator $\mathbf{1}[\cdot]$ equals one
when its condition is true and zero otherwise.

Over $N$ test images:

$$
\operatorname{RobustAccuracy}(\epsilon)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\hat y(x_i)=y_i
\land
\hat y(x_{\text{adv},i})=y_i
\right]
$$

Read this as:

> “Robust accuracy at epsilon is the fraction of all test examples that were
> correctly classified before the attack and remain correctly classified after
> the attack.”

An initially incorrect image is never counted as robust. If an attack happens
to turn an initially wrong prediction into the correct prediction, that does
not demonstrate robustness.

At $\epsilon=0$, the adversarial image equals the clean image, so robust
accuracy equals ordinary clean accuracy.

## 21. Results from the executed notebook

The current saved run produced:

| $\epsilon$ | Correct | Robust accuracy |
|---:|---:|---:|
| 0.00 | 9,921 / 10,000 | 99.21% |
| 0.05 | 9,571 / 10,000 | 95.71% |
| 0.10 | 8,575 / 10,000 | 85.75% |
| 0.15 | 6,625 / 10,000 | 66.25% |
| 0.20 | 3,982 / 10,000 | 39.82% |
| 0.25 | 2,042 / 10,000 | 20.42% |
| 0.30 | 981 / 10,000 | 9.81% |

As epsilon grows, the allowed perturbation grows:

$$
0.05<0.10<0.15<\cdots<0.30
$$

The attack can move farther in the loss-increasing direction, so accuracy
generally decreases.

The decrease does not have to be linear. Neural networks are nonlinear, the
Taylor equation is only a local approximation, pixels can hit clipping
boundaries, and different images cross classification boundaries at different
epsilon values.

## 22. Understanding the example grid

Each image title in the final plot has the form:

~~~text
original prediction -> adversarial prediction
~~~

For epsilon greater than zero, the notebook saves examples satisfying:

$$
\hat y(x)=y
\qquad\text{and}\qquad
\hat y(x_{\text{adv}})\neq y
$$

Read this as: “The original image was correctly classified, but the adversarial
image was misclassified.”

This avoids claiming success on an image the model already got wrong.

The epsilon-zero row is a baseline, so its titles show unchanged correct
predictions.

## 23. Hyperparameters in the notebook

| Name | Value | Purpose |
|---|---:|---|
| <code>BATCH_SIZE</code> | 64 | Images per training update |
| <code>TEST_BATCH_SIZE</code> | 1,000 | Images per clean evaluation batch |
| <code>ATTACK_BATCH_SIZE</code> | 256 | Images attacked together |
| <code>EPOCHS</code> | 14 | Complete passes over the training set |
| <code>LEARNING_RATE</code> | 1.0 | Initial Adadelta learning rate |
| <code>LR_GAMMA</code> | 0.7 | Per-epoch learning-rate multiplier |
| <code>MNIST_MEAN</code> | 0.1307 | Dataset normalization mean |
| <code>MNIST_STD</code> | 0.3081 | Dataset normalization standard deviation |
| <code>SEED</code> | 1 | Reproducible random initialization and shuffling |

The attack strengths are:

$$
\epsilon
\in
\{0,0.05,0.10,0.15,0.20,0.25,0.30\}
$$

Read this as: “Epsilon is selected from zero through zero point three in steps
of zero point zero five.”

## 24. Common mistakes and why they matter

### Updating the model during the attack

FGSM should not call <code>optimizer.step()</code>. The attack measures a fixed
model's vulnerability; updating it would change the object being measured.

### Forgetting <code>requires_grad</code> on the input

Without:

~~~python
pixels.requires_grad_(True)
~~~

PyTorch will not retain $\nabla_xJ$ in <code>pixels.grad</code>.

### Using training mode during the attack

If dropout remains active, the model changes randomly between passes. Calling
<code>model.eval()</code> gives a stable attack and evaluation.

### Adding pixel-scale epsilon to normalized data

An epsilon intended for $[0,1]$ pixels has a different numerical meaning after
normalization. The notebook explicitly attacks original-scale pixels and then
normalizes them.

### Forgetting to clip

Without clipping, some values could become negative or exceed one, producing
invalid image intensities.

### Counting initially wrong images as attack successes

An attack did not cause an error if the model was already wrong. The notebook
records successful examples only when the clean prediction was correct.

### Expecting epsilon to be a percentage of changed pixels

Epsilon limits how far **each** pixel may move. FGSM can change all 784 pixels.

### Confusing <code>model.zero_grad()</code> with an update

Clearing gradients changes no weights. Only an optimizer step changes the
parameters during this workflow.

## 25. The shortest useful summary

Normal training minimizes:

$$
J(\theta,x,y)
$$

by changing $\theta$:

$$
\theta\leftarrow\theta-\alpha\nabla_\theta J
$$

FGSM approximately maximizes the same loss by changing $x$:

$$
x_{\text{adv}}
=
\operatorname{clip}_{[0,1]}
\left(
x+\epsilon\operatorname{sign}(\nabla_xJ)
\right)
$$

The sign is used because, under:

$$
\|\delta\|_\infty\leq\epsilon
$$

the first-order loss is maximized by:

$$
\delta=\epsilon\operatorname{sign}(\nabla_xJ)
$$

In plain English:

> Train the model to make the correct answer likely. Freeze the trained model.
> Measure how every pixel would affect the loss. Move every pixel by the
> maximum allowed amount in its loss-increasing direction. Clip the result and
> classify it again.

## 26. Questions to test your understanding

1. Why does normal training subtract a gradient while FGSM adds a gradient
   sign?
2. What does the subscript $x$ mean in $\nabla_xJ$?
3. Why must <code>pixels.requires_grad_(True)</code> be set?
4. Why is epsilon applied before normalization?
5. Why does FGSM use <code>sign()</code> instead of the raw gradient?
6. Why is clipping necessary?
7. Why must the model be in evaluation mode during the attack?
8. Why do initially incorrect examples not count as robust?
9. What does $\|\delta\|_\infty\leq\epsilon$ say about each pixel?
10. Does calling <code>loss.backward()</code> automatically change the model?

<details>
<summary>Answers</summary>

1. Training wants to reduce loss, while the attack wants to increase it.
2. It means the loss is differentiated with respect to the input image.
3. Otherwise PyTorch does not retain an input gradient in
   <code>pixels.grad</code>.
4. The notebook defines epsilon in the understandable original $[0,1]$ pixel
   scale.
5. It chooses the loss-increasing endpoint of each pixel's allowed interval,
   maximizing the first-order loss under an infinity-norm constraint.
6. It keeps every perturbed pixel in the valid range from zero to one.
7. It disables dropout randomness and makes the attacked model stable.
8. Robustness means preserving a correct prediction; an already wrong
   prediction was not broken by the attack.
9. Every pixel's absolute change must be no greater than epsilon.
10. No. <code>backward()</code> calculates gradients; parameters change only
    when an optimizer performs a step.

</details>

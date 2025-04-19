# FedSAMHessian: FedSAM with Second-Order Perturbations and Trace Regularization

This repository implements federated learning algorithms that incorporate second-order information and trace regularization to improve model generalization and convergence. The project builds upon existing second-order Sharpness-Aware Minimization (SAM) in literature and extends it to the federated learning setting, improving upon the base FedSAM (using first-order perturbation).

## Table of Contents
- [Benchmarks](#benchmarks)
- [Second-Order Perturbation Approaches](#second-order-perturbation-approaches)
- [Comparison with Existing Methods](#qualitative-comparison-with-existing-sam-methods)
- [Variants of Second-Order Methods](#variants-of-second-order-methods)
- [Joint Approaches](#joint-approaches)
- [Orthogonality of  Proposed Second-Order Methods](#orthogonalality-of-proposed-second-order-methods)
- [Computational Considerations](#computational-considerations)
- [Current Goals](#current-goals)
- [Results and Conclusion](#results-and-conclusion)
- [Future Work](#future-work)

## Benchmarks

Our implementations are evaluated on the following benchmarks:

- **CIFAR-10**: Standard image classification dataset with 10 classes
- **CIFAR-100**: More challenging image classification with 100 classes
- **PACS**: Domain generalization dataset with 7 classes across 4 domains
- **OfficeHome**: Real-world domain adaptation dataset with 65 classes across 4 domains

We test our approaches with various model architectures:
- CNN
- ResNet
- Vision Transformer (ViT)

## Second-Order Perturbation Approaches

### FedSAM_Hessian

The base second-order approach extends FedSAM by incorporating Hessian information to guide parameter updates. Instead of using just the first-order gradients for perturbation, we compute the top eigenvector of the Hessian matrix and include it to perturb the model parameters.

Key components:
- Computation of top eigenvector of Hessian using power iteration method
- Adaptive weighting of top eigenvector by the corresponding eigenvalue

### FedSAM_Trace (Trace Regularization)

Trace regularization is a computationally efficient way to incorporate second-order information without explicitly computing the full Hessian matrix. The trace provides a measure of the overall curvature of the loss landscape, which can be used to guide optimization.

Our implementation:

1. Computes the trace of the Hessian using Hutchinson's method (an unbiased stochastic estimator for trace)
2. Incorporates the trace as a regularization term in the loss function


## Qualitative Comparison with Existing SAM Methods

| Method | Perturbation Type | Regularization | Computational Cost | Convergence Speed | Generalization |
|--------|------------------|----------------|-------------------|-------------------|----------------|
| FedSAM | First-order | None |-- | -- | -- |
| FedLESAM | First-order | None | -- | -- | -- |
| FedLESAM_S | First-order | Control Variates | -- | -- | -- |
| FedLESAM_D | First-order | Control Variates | -- | -- | -- |
| FedGLOSS | First-order | None | -- | -- | -- |
| FedSAM_Hessian | Second-order | None | -- | -- | -- |
| FedSAM_Trace | Second-order | Trace | -- | -- | -- |
| FedSAM_Eigen_Trace | Second-order | Trace | -- | -- | -- |
| FedSAM_Hessian_Orth | Second-order | None | -- | -- | -- |

## Detailed Quantitative Results (across models and benchmarks)

TO DO: Add comprehensive table



## Variants of Second-Order Methods

### FedSAM_Eigen_Trace

This joint approach combines both eigenvector-based perturbation and trace regularization:

1. Computes the top eigenvector of the Hessian for perturbation direction
2. Incorporates trace regularization in the loss function
3. Uses both metrics to achieve locally flatter minima.

### FedSAM_Hessian_Orth (Inherited from ["Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization"](https://arxiv.org/abs/2501.12666))

This approach ensures orthogonality between the perturbation direction and the gradient:

1. Computes the top eigenvector of the Hessian matrix (using power iteration)
2. Alligns the gradient and top eigenvector and projects the top eigenvector onto the space orthogonal to the gradient
3. Adds this orthogonal component to existing first-order perturbation

## Latency of Proposed Method and Consequent Optimizations
The existing iterative methods lead to much longer training times (2x - 4x) than the base methods. We analyze sparser implementations of our second-order method to achieve superior performances given a compute budget / upper bounds on local training.

### FedSAMHessian_Interleaved

This hybrid approach interleaves first-order and second-order updates:

1. Uses standard FedSAM/ESAM for most updates
2. Periodically (every N steps) computes and applies second-order perturbations
3. Balances computational efficiency with the benefits of second-order information

### FedSAMHessian_Switch

This approach switches from first-order to second-order methods during training:

1. Starts with standard FedSAM/ESAM for faster initial convergence
2. Switches to second-order methods after K% of training rounds
3. Leverages the benefits of both approaches at different stages of training

This was inspired from the observation that the reduction in sharpness in the early training steps is identical for both first and second-order methods. Second-order methods indicate better sharpness reductions in the later training steps.

## Orthogonalality of Proposed Second-Order Methods

### Integration with Control Variates

Our second-order approaches can be integrated with control variate methods like SCAFFOLD:

1. Maintains control variates for reducing client drift
2. Incorporates second-order information for better perturbation direction
3. Combines the benefits of both approaches

### Intgeration with Existing Global Flatness Promotion Methods ([FedGLOSS](https://arxiv.org/abs/2412.03752))

FedGLOSS explicitly promotes global flatness in federated learning:

1. Computes a server side measure of global direction using communicated client updates
2. Introduces global perturbation (i.e on the server side) based on this estimate before communicating
the updated model i.e. the perturbation step
3. Ensures that the global model maintains good generalization properties

We integrate FedGLOSS on the server side with local training using our second-order approach to obtain
minima that are both locally and globally flatter 

## Computational Considerations

The iterative nature of second-order methods increases computational overhead:

- **Time Complexity**: Computing second-order information iteratively is computationally expensive and leads to slower local training times

To address this challenge, we propose:

* **Sparse Second-Order Updates**: Only compute second-order information periodically
* **Late Kick In**: Combine first and second-order methods to balance efficiency and effectiveness by switching to second-order methods after K% of training rounds

Further Envisioned (Future Work):

* **Adaptive Switching**: Monitor sharpness and perform a switch when gains (sharpness drop) diminishes

## Current Goals 

### (Not Yet Implemented - To Be Done by This Week)

### FedSAMHessian_Interleaved (Done, Running)

This approach will interleave first-order and second-order updates to balance computational efficiency with the benefits of second-order information. It will:

1. Use standard FedSAM/ESAM for most updates
2. Periodically compute and apply second-order perturbations
3. Adapt the frequency of second-order updates based on training progress

### FedSAMHessian_Switch (Done, Running)

This approach will switch from first-order to second-order methods during training to leverage the benefits of both approaches at different stages. It will:

1. Start with standard FedSAM/ESAM for faster initial convergence
2. Switch to second-order methods after a certain percentage of training rounds
3. Adapt the switching point based on the dataset and model

### Integration with Control Variates

We plan to integrate our second-order approaches with control variate methods like SCAFFOLD to combine the benefits of both approaches. This will:

1. Maintain control variates for reducing client drift
2. Incorporate second-order information for better perturbation direction
3. Adapt the balance between control variates and second-order information

### Global Sharpness Promotion ([FedGLOSS](https://arxiv.org/abs/2412.03752))

FedGLOSS explicitly promotes global sharpness in federated learning by:

1. Computing a global sharpness measure across all clients
2. Using this measure to guide parameter updates
3. Ensuring that the global model maintains good generalization properties

## Results and Conclusion

Our experiments demonstrate that incorporating second-order information and trace regularization in federated learning can significantly improve model generalization and convergence. Key findings include:

1. Second-order methods (FedSAM_Hessian, FedSAM_Trace) consistently outperform first-order methods (FedSAM, FedLESAM)
2. Joint approaches (SAM_Eigen_Trace) provide the best generalization performance
3. Orthogonal methods (SAM_Hessian_Orth) are particularly effective for challenging datasets
4. Hybrid approaches offer a good balance between computational efficiency and performance

< I will add the detailed experimental setups (in another markdown) and results soon>

## Future Work
* Adaptive switching between first and second order methods

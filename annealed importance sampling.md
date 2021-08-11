Annealed Importance Sampling (AIS) is a strategy of importance sampling, applied to an extended state with a sequence of intermediate distributions ($ p_0 $ to $ p_1 $).


One popular choice for intermediate distributions is to use weighted geometric average of the target distribution p1. More specifically, this sequence is a design choice based on problem domain.

A series of Markov chain transition functions are applied to define the conditional probability distribution. transition operators were used to sequentially generate samples from the intermediate distributions from $ p_0 $ to $ p_1 $.

By chaining the importance weights for the jumps between the distributions throughout the sampling process, we can derive the importance weight, and estimate the ratio of particion functions:

$$
w^{(k)}=\frac{\tilde{p}_{\eta_{1}}\left(\boldsymbol{x}_{\eta_{1}}^{(k)}\right)}{\tilde{p}_{0}\left(\boldsymbol{x}_{\eta_{1}}^{(k)}\right)} \frac{\tilde{p}_{\eta_{2}}\left(\boldsymbol{x}_{\eta_{2}}^{(k)}\right)}{\tilde{p}_{\eta_{1}}\left(\boldsymbol{x}_{\eta_{2}}^{(k)}\right)} \cdots \frac{\tilde{p}_{1}\left(\boldsymbol{x}_{1}^{(k)}\right)}{\tilde{p}_{\eta_{n-1}}\left(\boldsymbol{x}_{\eta_{n}}^{(k)}\right)}
$$

$$
\frac{Z_{1}}{Z_{0}} \approx \frac{1}{K} \sum_{k=1}^{K} w^{(k)}
$$

AIS is the most common way of estimating partition function for undirected probabilistic models, such as a trained model of Restricted Boltzmann Machine.




The AIS sampling strategy is to generate samples from p0 and 











These transitions may be also constructed as Metropolis-Hastings, Gibbs, including methods involving multiple passes through all the random variables or other kinds of iterations.



of transitioning to x given we are currently at x.



ny Markov chain Monte Carlo method(e.g., 








premises:
Ensure Understanding:

what is transition operator
what is undirected probabilistic models
what is Boltzmann machine
what is simple importance sampling
what is intermediate distributions

this is not a sampling methos, it is a approximation method
if we can efficiently draw sample from the distribution in monte carlo estimation

importance sampling can help determine the expectation mean of function g by looking at the ratio between two functions g and reference function f

what is normaliazed likelihood of data
- it would help us evaluate the model, monitor training performance, and compare models
what is partition function
why is it important to estimate partition function

how to estimate partition function
- we need this to compute normaliazed likelihood of data
- this is a challenging task when we do this for complex distributions over high-dimensional spaces because... there are two main strategies to cope with the challenges, annealed importance sampling and bridge sampling.
how to avoid computing intractable partition function

=== main ===

Annealed (退火) importance sampling (AIS) was first discovered by jarzynski and independently by neal. it is currently the most common way of estimating the partition function for undirected probabilistic models

how does Annealed (退火) importance sampling (AIS) work
- it can bridge the gap between P1 and P0, where a little overlap between the 2 distributions (i.e. D-KL(p0||p1) is large), with intermediate distributions. 


Use tach of the factors (Znj+1/Znj) to obtain an estimate of Z1/Z0

popular choice for the intermediate distributions is touse the weighted geometric average of the target distribution and the starting proposal distribution

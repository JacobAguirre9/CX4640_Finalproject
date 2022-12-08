---
Name: Jacob Aguirre
Topic: Stochastic Optimization
Title: A case study of Stochastic Optimization techniques
----

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Algorithms](#algorithms)
- [Discussion](#discussion)
- [References](#references)

# Introduction

Stochastic optimization is a method for finding the minimum or maximum of a function. As contrast to deterministic algorithms, which always choose the same course of action given the same input, this kind of optimization algorithm uses randomness to reach decisions. Stochastic optimization techniques operate by incrementally altering the input to enhance the output. The method randomly chooses a point in the input space at each step and evaluates the function there. The algorithm modifies the input to enhance the output based on the evaluation's findings. Until the function achieves its minimum or maximum value, this process is repeated.

One advantage of stochastic optimization is that it can avoid getting stuck in local minima or maxima, which are points where the function has a minimum or maximum value but is not the global minimum or maximum. Deterministic algorithms, on the other hand, can easily get stuck in local minima or maxima, making it difficult for them to find the global minimum or maximum. Another advantage of stochastic optimization is that it can be parallelized easily, meaning that it can be run on multiple computers or processors at the same time. This can make the algorithm faster and more efficient, especially for large and complex optimization problems.

Stochastic optimization is used in a wide range of applications, including machine learning, finance, engineering, and operations research. It is a powerful tool for solving optimization problems and can be an effective alternative to deterministic algorithms in many situations. We will discuss the advantages, disadvantages, applicable case studies, and more in this paper. I will motivate some of the questions and materials discussed below by discussing my own research work on addictive substances and smoking that is conducted at Oak Ridge National Laboratories (ORNL) and ISyE department at Georgia Tech.

# Background

Typically, a stochastic optimization problem (SOP) is composed of three main ideas: We have the state variable, which defines the information for the problem, the uncertainty itself (which can take many forms) and the decisions that the agent(s) take within the problem. Before we dive into any algorithms or methods for solving these SOPs, let's look at the state variable. Following the methodologies of Puterman (2014) and Powell (2021), we define the state variable $S_t$ to be consisted of the following: 

1. $R_t$: The Physical state. This physical state consists of all the physical information known at time $t$. Physical states restrict the decisions we can make in some way. The location on a network determines the decisions we can make. The decisions affect the physical state, either directly or indirectly. For example, the amount of smoking stock that a person has at time $t-1$ will affect their decisions and actions to make a quit attempt at time $t,t+1$.

2. $I_t$: The Information state. We consider this state to include information that affects the behavior of a problem. For example, consider the paper on Class-Ordered Monotone Policies by Garcia et al. (2022) in Optimization Online. In this paper, the authors studied the optimal treatments for Hypertension patients. Contained within the information state is adherence level (by both patient to take their medications, and practicioner to trust the recommendation system), information about the drugs, and risk levels. We know this information, and it is captured perfectly. Some consider this to be information that is _observable_.

3. $B_t$: The belief state. The belief state is the information that specifies a probability distribution describing an unknown parameter. Consider that this could be the patients response to a medication, the patients response to being told their diagnosis and updated risk beliefs, etc. The key component and contrast with the information state is that this information _is not observed perfectly_. For example, the belief state plays a huge role in Partially-Observable Markov Decision Processes where the state space is not entirely visible for the agent and they only have beliefs. 

We will now motivate this problem with decisions, and different types of decisions that are possible. 

1. Consider that decisions could be as simple as binary $\mathcal{X}=\{0,1\}$. This could be interpreted as either the patient takes the medication or doesn't, we either buy the stock or sell, etc. 
2. Another common form of decision is continuous vector. Thus, we would have $\mathcal{X}=\mathbb{R}^n$ where we have an $n$-dimensional vector. This would more commonly be seen within the control community or system dynamics, and not as often within the Operations Research community.
3. Finally, although there are countless other ways to quantify a decision, let's consider categorical. Say we have a set $\mathcal{X}=1,...,N$ where $N$ is the number of available drugs for a patient $i\in I$ to try. So, for each object $n\in\mathcal{X}$ we have an associated list of attributes $a=a_1,a_2,...,a_n$ which describe each object in the category.

Finally, we will provide a brief insight into different types of uncertainty. 
1. Distributions. When we use probability distributions, we can draw on a libary of distributions such as normal, exponential, Poisson and uniform (to name but a few). For counting processes especially, we will use Poisson processes.
2. Finally, consider Distribution-free uncertainty. In situations such as this, we must depend on the exogenous information we observe after making decisions and updating states. We _do not have_ a formal probability model in situations like this.

# Algorithms

The term "stochastic optimization" (SO) refers to optimization techniques that produce and employ random variables. The random variables for stochastic issues are incorporated into the formulation of the optimization problem, which includes random objective functions or random constraints. Sometimes these random constraints look like stochastic dominance, both first-order and second-order. These problems, while not studied in-depth in Operations Research, are well-studied microtheory economic principles. This further highlights the usage and importance of modeling stochastic problems and how the field expands into multiple other communities. Furthermore, the use of random iterations is also a feature of stochastic optimization techniques. This is similar to where we consider the field of robust optimization, where we're optimizing for the best outcome given the worst possible scenario $\omega\in\Omega$. For more into this narrow topic, see Schaefer et al (2022).

When setting up an optimization problem, whether stochastic or deterministic, we're typically seeking to find the max or min of some metric. For example, this could be minimizing costs in a firm, maximizing revenue, etc. We will look at three main policies and algorithms for solving SOPs. 
1. Sample Average Approximation 
2. Stochastic Gradient Descent
3. Policy Iteration methods

### Sample Average Approximation

The sample average approximation (SAA) method use Monte Carlo simulation to address stochastic optimization constraints. Suppose we're given a probability space $\Omega$ with countable measure and we can describe an event as $\omega\in\Omega.$ In economics, this is called a "lottery" over a probability space, where $\sum\limits_{i=1}^n\omega_i=1$. Furthermore, suppose we have a random vector, call it $\zeta$, with $\zeta_1,\zeta_2,...\zeta_n$ representing _replications_ of the random vector. We will say that this follows an independent and identically distributed sample. So the sample average can be described as, 

$$\hat{f}(x) = \frac{1}{n}\sum_{i=1}^{n} f(x_i)$$

This formula represents the sample average approximation of a function $f(x)$ at a point $x$ based on a set of $n$ sample points $x_1, x_2, ..., x_n$. 

The sample average approximation is a simple and computationally efficient method for estimating the value of a function at a particular point based on a set of sample data. One of the main advantages of this method is that it is easy to implement and requires only basic mathematical operations, making it a good choice for applications where computational resources are limited or time is of the essence.

However, the sample average approximation also has some limitations. One major disadvantage is that it can be highly sensitive to the choice of sample points, which can lead to significant errors in the estimated function values if the sample points are not carefully chosen. Additionally, the sample average approximation is only an estimate of the true function value, and it is generally not as accurate as other more sophisticated approximation methods, such as polynomial regression or spline interpolation. 

We will now prove that this estimator is consistent, a common tool and metric widely used in numerical analysis and econometrics. 

To prove the consistency of sample average approximation, we must first define some notation. Let the random variable $X$ be distributed according to the probability distribution $P$, and let the function $f$ be a real-valued function defined on the sample space of $X$. The expected value of $f$ with respect to $P$ is denoted as $E[f(X)]$.

To prove the consistency of sample average approximation, we must show that as the size of the sample used to approximate the expected value of $f$ increases, the error between the sample average and the true expected value decreases. This can be formalized as follows:

Suppose that we have a sequence of samples of increasing size, denoted as $S_1$, $S_2$, $S_3$, $\dots$, where $S_n$ is a sample of size $n$ drawn from the distribution $P$. We can then define the error between the sample average and the true expected value at each step in the sequence as follows:

$Err(S_n) = \left|E[f(X)] - \operatorname{avg}(f(S))\right|$

where $\operatorname{avg}(f(S))$ is the average value of the function $f$ over the sample $S$.

To prove that the error decreases as the size of the sample increases, we must show that the sequence $Err(S_n)$ is a non-increasing sequence. That is, we must show that $Err(S_{n+1}) \leq Err(S_n)$ for all $n$.

To do this, we can use the triangle inequality to expand the definition of $Err(S_{n+1})$ as follows:

$Err(S_{n+1}) = \left|E[f(X)] - \operatorname{avg}(f(S))\right| = \left|E[f(X)] - \left(\operatorname{avg}(f(S)) - \left(f(x_{n+1}) - \operatorname{avg}(f(S))\right)\right)\right|$

where $x_{n+1}$ is the $(n+1)$th element of the sample $S$.

Applying the triangle inequality, we have:

$Err(S_{n+1}) \leq \left|E[f(X)] - \operatorname{avg}(f(S))\right| + \left|\operatorname{avg}(f(S)) - \left(f(x_{n+1}) - \operatorname{avg}(f(S))\right)\right|$

$= Err(S_n) + \left|\operatorname{avg}(f(S)) - \left(f(x_{n+1}) - \operatorname{avg}(f(S))\right)\right|$

Since $\left|\operatorname{avg}(f(S)) - \left(f(x_{n+1}) - \operatorname{avg}(f(S))\right)\right|$ is always a non-negative quantity, we have:

$Err(S_{n+1}) \leq Err(S_n)$

This shows that the error between the sample average and the true expected value decreases as the size of the sample increases, and therefore the sample average approximation method is consistent.

In other words, as the size of the sample increases, the sample average converges to the true expected value of the function $f$ with respect to the probability distribution $P$.

Overall, the sample average approximation is a useful tool for approximating the value of a function at a particular point, but it should not be relied upon for highly accurate results in all cases. It is best used in situations where computational efficiency and simplicity are more important than the absolute accuracy of the approximation.

### Stochastic Gradient Descent

Stochastic gradient descent (SGD) is an iterative optimization algorithm that is commonly used in machine learning and OR and other fields for finding the minimum of a function. It works by iteratively updating the parameters of the function using the gradient of the cost function calculated from a small, randomly selected subset of the data (called a "mini-batch").

At each iteration, the algorithm computes the gradient of the cost function with respect to the parameters using a randomly selected mini-batch of data. The parameters are then updated in the direction of the negative gradient, using a pre-specified learning rate that determines the size of the step taken. This process is repeated until the algorithm converges to a minimum of the function, which is typically determined by a predefined stopping criterion.

One of the key advantages of stochastic gradient descent is that it is computationally efficient and can be applied to very large datasets. It also has the ability to escape from local minima and converge to the global minimum of the function, which is not always possible with other optimization algorithms. 

$$\theta_t = \theta_{t-1} - \alpha \nabla_{\theta} J(\theta_{t-1})$$

This formula represents the update step in stochastic gradient descent, where $\theta_t$ is the current value of the parameters, $\theta_{t-1}$ is the previous value of the parameters, $\alpha$ is the learning rate, and $\nabla_{\theta} J(\theta_{t-1})$ is the gradient of the cost function $J$ with respect to the parameters $\theta$ at the previous step. Stochastic gradient descent is an iterative optimization algorithm that is commonly used in machine learning and other fields for finding the minimum of a function. It works by iteratively updating the parameters of the function using the gradient of the cost function calculated from a small, randomly selected subset of the data (called a "mini-batch"). This approach allows the algorithm to converge to the minimum more quickly than other methods, such as batch gradient descent, but it can also be less stable and may not always converge to the global minimum of the function.

Stochastic gradient descent is often a good choice when the dataset is very large and it is not feasible to use other optimization algorithms that require the computation of the gradient over the entire dataset. In such cases, the use of mini-batches can significantly reduce the computational cost of the algorithm and allow it to converge to the minimum of the function more quickly.

Another situation where stochastic gradient descent may be optimal is when the cost function has a large number of local minima and the global minimum is difficult to find using other optimization algorithms. In such cases, the ability of stochastic gradient descent to escape from local minima and converge to the global minimum can be very useful. Again, we notice how SOPs tend to deal with finding minimums and maximums of random functions.

One of the main disadvantages of stochastic gradient descent is that it can be less stable than other optimization algorithms, such as batch gradient descent. This is because the use of randomly selected mini-batches can cause the algorithm to fluctuate more and may result in suboptimal solutions. Additionally, the use of a fixed learning rate can cause the algorithm to converge to a suboptimal solution if the learning rate is not carefully chosen, and it may be difficult to determine the optimal learning rate for a given problem.

Another disadvantage of stochastic gradient descent is that it may not always converge to the global minimum of the function, unlike some other optimization algorithms. This can be especially problematic in high-dimensional problems, where the cost function may have many local minima and the algorithm may get stuck in a suboptimal solution. High dimensional problems may deal with SGD better if they first have nonlinear dimension reduction techniques applied first. 

### Value Policy Iteration

Policy iteration is a method for solving Markov decision processes (MDPs), which are mathematical models used in decision-making and control problems. In policy iteration, an initial policy is chosen and then used to evaluate the value function, which represents the expected long-term reward of following the policy. The policy is then improved based on the value function, and the process is repeated until the policy converges to the optimal policy.

Policy iteration has several advantages over other methods for solving MDPs. It is relatively simple and easy to implement, and it is guaranteed to converge to the optimal policy under certain conditions. Additionally, policy iteration can be easily parallelized, which makes it a good choice for large or complex problems. While the problems students or researchers in OR might be relatively easy to solve without using parallel computing resources, these resources are needed heavily at places such as ORNL when dealing with high dimensional problems that could take days to solve. We will now look at one famous equation in reinforcement learning and value policy iteration for SOPs.

$$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) \sum_{s' \in \mathcal{S}} p(s' | s, a) [r(s, a, s') + \gamma V^{\pi}(s')]$$

This equation describes the value of a state $s$ under a particular policy $\pi$ in a Markov decision process (MDP). The value of the state is the expected long-term reward of following the policy, taking into account the probabilities of transitioning to other states and the rewards associated with those transitions. The discount factor $\gamma$ is used to balance the importance of immediate rewards versus long-term rewards.

Bellman's equation has many important applications in fields such as reinforcement learning, operations research, and artificial intelligence. It provides a fundamental framework for solving MDPs, and it has been used to develop many powerful algorithms for solving complex decision-making and control problems. For applicable real-world problems solved using these methods, see Alagoz (2010), Ajayi (2021), and Alagoz (2012).

Finally, we will look at one more equation that is known for Value policy iteration. 

$$\begin{aligned} V_{k+1}(s) &= \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} p(s' | s, a) [r(s, a, s') + \gamma V_{k}(s')] \ \pi_{k+1}(s) &= \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} p(s' | s, a) [r(s, a, s') + \gamma V_{k}(s')] \end{aligned}$$

This formula represents the update step in value policy iteration, where $V_k$ is the current estimate of the value function, $V_{k+1}$ is the updated estimate of the value function, $\pi_k$ is the current policy, and $\pi_{k+1}$ is the updated policy. In each iteration of the algorithm, the value function is updated based on the current policy, and then the policy is updated based on the updated value function. This process is repeated until the policy converges to the optimal policy.




# Discussion

In conclusion, stochastic optimization problems are a type of mathematical optimization problem that involve uncertainty or randomness. These problems can be challenging to solve due to the presence of randomness, but a variety of techniques, such as stochastic gradient descent and value policy iteration, can be used to find good solutions. Understanding and solving these types of problems is important in a variety of fields, including machine learning and operations research. 

If I still haven't convinced you of why we need stochastic optimization, consider the following. Stochastic optimization is needed because many real-world problems involve uncertainty or randomness. You literally can't make real-world decisions without uncertainty being there. For example, in finance, stock prices are constantly fluctuating and are difficult to predict. In logistics, demand for goods and services can be difficult to forecast. In these and other similar situations, using deterministic optimization techniques, which assume that all variables are known and fixed, can produce suboptimal solutions. Stochastic optimization, on the other hand, allows us to take uncertainty into account and find solutions that are more robust and better able to handle unpredictable events. This is especially important in situations where the cost of making a suboptimal decision can be high.

In my own research field within medical decision making, SO is still widely used and important. In medical decision making, stochastic optimization can be used to identify the best course of action in situations where there is uncertainty or randomness. For example, a doctor may need to decide which treatment to recommend to a patient based on the patient's individual characteristics, such as age, medical history, and current condition. However, the effectiveness of the treatment may vary from one patient to another, and there may be multiple factors that influence its success. In this situation, stochastic optimization can be used to find the treatment that is most likely to be effective for a given patient, based on the available data and the uncertainty involved. This can help doctors make more informed and better decisions, and ultimately lead to better outcomes for patients.

# References
1. Tian, Z., Han, W., & Powell, W. B. (2022). Adaptive learning of drug quality and optimization of patient recruitment for clinical trials with dropouts. Manufacturing & Service Operations Management, 24(1), 580-599.https://doi.org/10.1287/msom.2020.0936
2. Puterman, M. L. (2014). Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons.
3. Garcia, G. G. P., Steimle, L. N., Marrero, W. J., & Sussman, J. B. An Analysis of Structured Optimal Policies for Hypertension Treatment Planning: The Tradeoff Between Optimality and Interpretability. https://optimization-online.org/?p=17279
4. Alagoz, O., Hsu, H., Schaefer, A. J., & Roberts, M. S. (2010). Markov decision processes: a tool for sequential decision making under uncertainty. Medical Decision Making, 30(4), 474-483. [HTML] from nih.gov
5. Siebert, U., Alagoz, O., Bayoumi, A. M., Jahn, B., Owens, D. K., Cohen, D. J., & Kuntz, K. M. (2012). State-transition modeling: a report of the ISPOR-SMDM modeling good research practices task force–3. Medical Decision Making, 32(5), 690-700.
6. Ajayi, T., Lee, T., & Schaefer, A. J. (2022). Objective selection for cancer treatment: an inverse optimization approach. Operations Research.[PDF] from optimization-online.org
7. Ajayi, T., Hosseinian, S., Schaefer, A. J., & Fuller, C. D. (2021). Combination Chemotherapy Optimization with Discrete Dosing. arXiv preprint arXiv:2111.02000.

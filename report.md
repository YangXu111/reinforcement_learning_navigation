# Project 1: Navigation Report

In this project, I implemented four DQN learning algorithms: DQN with experience replay, double DQN, DQN with prioritized experience replay, dueling DQN. I take the DQN with experience replay as the baseline algorithm to gauge other model's performance. The DQN algorithm is also applied to the pixel environment.

# Results
## DQN with Experience Replay
### Algorithm
The learning algorithm can be summarized as follows (modified from [Deep Q-Network (DQN)-II](https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c))

Initialize network Q  
Initialize target network Q'  
Initialize experince replay memory D  
Initialize the Agent  
Repeat until the max episodes:  
&emsp; Setting a new &epsilon;  
&emsp; Given state s, choose an action a using &epsilon;-greedy policy(Q)  
&emsp; Observe reward r and next state s'   
&emsp; Store transition (s, a, r, s', done) in the experience replay D  
&emsp; &emsp; If the number of past episodes is a multiple of K and if D has more than minibatch size experiences:  
&emsp; &emsp; Sample a minibatch N of experiences from D  
&emsp; &emsp; For every experience (s<sub>i</sub>, a<sub>i</sub>, r<sub>i</sub>, s<sub>i</sub>', done) in the minibatch:  
&emsp; &emsp; &emsp; If done then  
&emsp; &emsp; &emsp; &emsp; y<sub>i</sub> = r<sub>i</sub>  
&emsp; &emsp; &emsp; else  
&emsp; &emsp; &emsp; &emsp; y<sub>i</sub> = r<sub>i</sub> + gamma * max<sub>a'</sub> Q'(s<sub>i</sub>', a')  
&emsp; &emsp; &emsp; end  
&emsp; &emsp; end  
&emsp; &emsp; Calculate the L = 1/N &sum;<sub>i=0</sub><sup>N-1</sup>(y<sub>i</sub> - Q(s<sub>i</sub>, a))<sup>2</sup>    
&emsp; &emsp; Update Q with SGD to minimize L  
&emsp; &emsp; Soft update Q' with the weights of Q with parameter &tau;  
&emsp; &emsp; end  
end

### Network Structure and Hyperparameters
The network structure is
![DQN network](plot\dqn_network.png)

The hyperparameters are  
```
"hidden_layer_size": 128,
"random_seed": 2,
"learning_rate": 0.01,
"tau": 0.1,
"gamma": 0.99,
"memory_size": 1000,
"update_interval": 4,
"sample_size": 64,
"num_episodes": 2000,
"epsilon_begin": 1,
"epsilon_stable": 0.01,
"epsilon_decay": 0.9977  
```

### Results
![DQN with Experience Replay plot](plot\episode_score_DQN_20210224_104505.png)  
The agent reached the score of 13 for consecutive 100 episodes after about 1300 episodes. After that the agent's performance stablized around score 14 and has gone as high as above 25.
The following is a screenshot for the trained agent. The agent is moving smoothly towards yellow bananas while evading blue ones.
![DQN with Experience Replay ScreenRecord](plot\dqn.gif)

## Double DQN with Experience Replay
### Algorithm
Double DQN is similar to Regular DQN except that in the TD target the maximizing action is determined by the target network instead of the current network.

Initialize network Q  
Initialize target network Q'  
Initialize experince replay memory D  
Initialize the Agent  
Repeat until the max episodes:  
&emsp; Setting a new &epsilon;  
&emsp; Given state s, choose an action a using &epsilon;-greedy policy(Q)  
&emsp; Observe reward r and next state s'   
&emsp; Store transition (s, a, r, s', done) in the experience replay D  
&emsp; &emsp; If the number of past episodes is a multiple of K and if D has more than minibatch size experiences:  
&emsp; &emsp; Sample a minibatch N of experiences from D  
&emsp; &emsp; For every experience (s<sub>i</sub>, a<sub>i</sub>, r<sub>i</sub>, s<sub>i</sub>', done) in the minibatch:  
&emsp; &emsp; &emsp; If done then  
&emsp; &emsp; &emsp; &emsp; y<sub>i</sub> = r<sub>i</sub>  
&emsp; &emsp; &emsp; else  
&emsp; &emsp; &emsp; &emsp; y<sub>i</sub> = r<sub>i</sub> + gamma * Q'(s<sub>i</sub>', argmax<sub>a'</sub>Q(s<sub>i</sub>', a')) (Note the distinction with DQN)   
&emsp; &emsp; &emsp; end  
&emsp; &emsp; end  
&emsp; &emsp; Calculate the L = 1/N &sum;<sub>i=0</sub><sup>N-1</sup>(y<sub>i</sub> - Q(s<sub>i</sub>, a))<sup>2</sup>    
&emsp; &emsp; Update Q with SGD to minimize L  
&emsp; &emsp; Soft update Q' with the weights of Q with parameter &tau;  
&emsp; &emsp; end  
end

### Network Structure and Hyperparameters
The network structure is the same as DQN and so are the hyperparameters.

### Results
![Double DQN with Experience Replay plot](plot\episode_score_Double_DQN_20210224_121338.png)


The agent reached the score of 13 for consecutive 100 episodes after about 1200 episodes, earlier than the DQN agent. Simlar to DQN, after that the agent's performance stablized around score 15 and has gone as high as above 25.
The following is a screenshot for the trained agent. The agent is moving smoothly towards yellow bananas while evading blue ones.
![DQN with Experience Replay ScreenRecord](plot\double_dqn.gif)

## DQN with Prioritized Experience Replay
### Algorithm
The learning algorithm can be summarized as follows:

Initialize network Q  
Initialize target network Q'  
Initialize experince replay memory D with p = 1   
Initialize the Agent  
Repeat until the max episodes:  
&emsp; Setting a new &epsilon;  
&emsp; Given state s, choose an action a using &epsilon;-greedy policy(Q)  
&emsp; Observe reward r and next state s'   
&emsp; Store transition (s, a, r, s', done) in the experience replay D with maximal priority max<sub>i&isin;D</sub> p<sub>i</sub>  
&emsp; &emsp; If the number of past episodes is a multiple of K and if D has more than minibatch size experiences:  
&emsp; &emsp; Sample a minibatch N of experiences from D with probability for experience j as p(j) ~ p<sub>j</sub><sup>&alpha;</sup>/&sum;<sub>i</sub>p<sub>i</sub><sup>&alpha;</sup>  
&emsp; &emsp; Calculate the importance-sampling weights w<sub>j</sub> = (p<sub>j</sub>)<sup>-&beta;</sup>/max<sub>i&isin;D</sub> p<sub>i</sub>  
&emsp; &emsp; Update transition priority p<sub>j</sub> = |&delta;<sub>j</sub>|    
&emsp; &emsp; For every experience (s<sub>i</sub>, a<sub>i</sub>, r<sub>i</sub>, s<sub>i</sub>', done) in the minibatch:  
&emsp; &emsp; &emsp; If done then  
&emsp; &emsp; &emsp; &emsp; y<sub>i</sub> = r<sub>i</sub>  
&emsp; &emsp; &emsp; else  
&emsp; &emsp; &emsp; &emsp; y<sub>i</sub> = r<sub>i</sub> + gamma * max<sub>a'</sub> Q(s<sub>i</sub>', a')  
&emsp; &emsp; &emsp; end  
&emsp; &emsp; end  
&emsp; &emsp; Calculate TD error &delta;<sub>j</sub> = y<sub>i</sub> - Q(s<sub>i</sub>, a)   
&emsp; &emsp; Calculate the L = 1/N &sum;<sub>i=0</sub><sup>N-1</sup>w<sub>i</sub> &delta;<sub>j</sub><sup>2</sup>    
&emsp; &emsp; Update Q with SGD to minimize L. Note that only Q(s<sub>i</sub>, a) needs to calculate gradient.  
&emsp; &emsp; Soft update Q' with the weights of Q with parameter &tau;  
&emsp; &emsp; end  
end

### Network Structure and Hyperparameters
The network structure is the same as DQN.

The hyperparameters are  
```
"hidden_layer_size": 128,
"random_seed": 2,
"learning_rate": 0.01,
"tau": 0.1,
"gamma": 0.99,
"memory_size": 1000,
"update_interval": 4,
"sample_size": 64,
"num_episodes": 2000,
"epsilon_begin": 1,
"epsilon_decay": 0.9977,
"epsilon_stable": 0.01,
"small_const": 0.01,
"alpha": 0.6,
"beta_begin": 0.4,
"beta_increase": 0.0003,
"beta_stable": 1
```
Note that the decay rate is a lot higher than other learning algorithms. This is because I found when the lower decay rate was applied the performance did not give comparible performance to other algorithms.

### Results
![DQN with Prioritized Experience Replay plot](plot\episode_score_Prioritized_Experience_Replay_20210225_114630.png)  
The agent reached the score of 13 for consecutive 100 episodes after about 1700 episodes. After that the agent's performance stablized around score 13 and has gone as high as 23.
The following is a screenshot for the trained agent. The agent is moving smoothly for yellow bananas picking.
![DQN with Experience Replay ScreenRecord](plot\prioritized_experience_replay_larger_beta.gif)

## Dueling DQN with Experience Replay
### Algorithm
The learning algorithm is the same as DQN with Experience Replay but with the dueling network.


### Network Structure and Hyperparameters
The network structure is
![Dueling DQN network](plot\dueling_dqn_network.png)

The hyperparameters are the same as for DQN.  


### Results
![Dueling DQN with Experience Replay plot](plot\episode_score_Dueling_DQN_20210225_170540.png)  
The agent reached the score of 13 for consecutive 100 episodes after about 1500 episodes. After that the agent's performance stablized around score 14 and has gone as high as above 25.
The following is a screenshot for the trained agent. The agent is moving smoothly towards yellow bananas.
![DQN with Experience Replay ScreenRecord](plot\dueling_dqn.gif)

## DQN with Experience Replay on Pixels
### Algorithm
The algorithm is the same as DQN except that a CNN network is used to process the pixel inputs.

### Network Structure and Hyperparameters
The network structure is
![Pixel DQN network](plot\pixel_dqn_network.png)

The hyperparameters are  
```
"hidden_layer_size": 128,
"random_seed": 2,
"learning_rate": 0.01,
"tau": 0.1,
"gamma": 0.99,
"memory_size": 1000,
"update_interval": 4,
"sample_size": 64,
"num_episodes": 6000,
"epsilon_begin": 1,
"epsilon_stable": 0.01,
"epsilon_decay": 0.999 
```

### Results
![DQN with Experience Replay plot](plot\episode_score_pixel_DQN_20210228_234205.png)
During training within the pixel environment a memory leak problem occurred that the Unity environment took in more and more memory until almost all available memory and then exited. However, the agent needs more episodes for training in this environment. As such the agent did not reach averagely score 13 but still showed progress in performance improvement. Starting from 0, the agent was able to score 5 in average at around 3000 episodes. If more training episodes were possible, the agent was expected to generate more sound performance.


# Summary and Ideas for Future Work
The following figure illustrates the 100 episode moving average for each algorithm for the non-pixel environment.
![Summary](plot\summary.png)  
We observe that DQN, Double DQN and Dueling DQN have similar performance: reach average score of 13 after 1500 episodes. The DQN with prioritized experience replay does not perform in par with other algorithms but still stablizes around score 12.
In this project training an agent takes around one hour under GPU for each learning algorithm, it is very time consuming for hyperparameter tuning, thus I use the same hyperparameter across all the algorithms whenever possible for the sake of comparison. For the future work I would explore the following  
1. Bayesian hyperparameter tuning for each learning algorithm and compare their performance under respective optimal hyperparameters.
2. Studying the effect of each hyperparameter, particularly the network structure for each learning algorithm.
3. Fixing the memory leak problem with the pixel environment and training for 6000 episodes.


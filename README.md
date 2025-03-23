# This project is part of my Conversational AI course teached by Pr. Mirco Ravanelli at Concordia University. 

The goal of this project is to explore neural networks that grow during training. While traditional training involves a fixed number of parameters in a neural network, 
this project focuses on dynamically increasing the parameters over time, similar to the adaptive nature of our brain. 

Growing neural networks offers several potential benefits. 
- Firstly, it potentially enables strong regularization effects by initially learning simpler functions and gradually incorporating 
more complex ones during subsequent training stages.
- Secondly, this approach improves computational efficiency since the network starts small and gradually expands. 
- Thirdly, growing neural networks naturally align with the paradigm of continual learning, allowing the acquisition of new tasks without forgetting previously learned ones.

The goal of this project is to implement some strategies for growing neural networks and test them on speech processing tasks.

Students are encouraged to follow these steps:

**Literature Review:**
- Conduct a detailed review of the literature to develop a comprehensive understanding of neural network growth strategies. Focus on key methodologies, challenges, and current state-of-the-art approaches.

**Baseline Implementation:**
- Start with a simple implementation of Net2Net to grow a basic MLP using the MNIST dataset as a toy task. This will help you familiarize yourself with the concept of growing neural networks in a controlled, straightforward setting.

**Transition to Speech Processing Tasks:**
- After gaining familiarity with MNIST, you can advance to more complex tasks such as speech recognition using the TIMIT dataset (available [here](https://www.dropbox.com/scl/fi/t6ql1ef4odthpdi5dxd6r/TIMIT.tar.gz?rlkey=j8xyxnc2wk2saaj2reej0yl4a&e=1&st=ne4iprly&dl=0)).
- Here you need to develop strategies to grow a neural network beyond simple MLPs and deal with more realistic and challenging tasks.
- Use the TIMIT ASR SpeechBrain recipe (available [here](https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT/ASR/CTC) as a starting point.)

**Comparison of Approaches:**
- Explore and implement different growing strategies and algorithms.
- Compare their effectiveness, focusing on aspects such as performance improvement, computational cost, and scalability.
- Investigate critical questions, such as determining the optimal moment to grow the network.

**Optional for a Better Grade:**
- Propose novel strategies for growing neural networks, going beyond existing methods in the literature.
- **Growing Scheduler:** Develop heuristics and solutions for an effective growing scheduler. This scheduler should:
  - Determine the optimal timing for a growing step.
  - Decide the appropriate number of neurons or layers to add at each step.


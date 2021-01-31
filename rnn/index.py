# https://www.youtube.com/watch?v=LHXXI4-IEns
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/


# One to One:
# Examples: Vanilla NN Image classification

# Many to one:
# Examples:
# 1) Sentiment analysis (opinion mining or emotion AI)
# 2) Action/Activity recognition from Sequence of video frame

# Many to Many:
# Example
# 1) Machine translation
# 2) Video Classification on frae level

# One to Many: Image captioning

import numpy as np
#nn_output=np.random.randint(low=1, high=10,size=20)
size=20
nn_output=10*np.random.randn(size)
#print("network output:",nn_output)
# temprature > 1 --> "soft" all probabilities will same value
# temprature < 1 --> "hard"  the one with the max value will have much higher probability,

temprature=1
exp_nn_output=np.exp(nn_output/temprature)
#print(exp_nn_output)
nn_output_softmax_prbabilities=exp_nn_output/np.sum(exp_nn_output)
#print(nn_output_softmax_prbabilities)
#print(np.sum(nn_output_softmax_prbabilities))
sample = np.random.multinomial(1, nn_output_softmax_prbabilities, 1)
index=np.where(sample==1)[1]
print("picked value from nn output number on index:",index)
#print(sample)
print("and value is:",nn_output[index])

print("inde of max value of nn:",np.argmax(nn_output))
print("and value is:",np.max(nn_output))

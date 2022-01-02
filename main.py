import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

slot_arms = [2, 0, 0.2, -2, -1, 0.8]
len_slot_arms = len(slot_arms)
def findReward(arm):
  result = np.random.randn(1)
  if result > arm:
    # returns a positive reward
    return 1
  else:
    # returns a negative reward
    return -1
tf.reset_default_graph()
weights = tf.Variable(tf.ones([len_slot_arms]))
chosen_action = tf.argmax(weights, 0)

reward_holder = tf.placeholder(shape=[1], dtype=tf.int32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
reward_holder = tf.cast(reward_holder, tf.float32)
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# Loss = -log(weight for action)*A

total_episodes = 1000
total_reward = np.zeros(len_slot_arms) # output reward array
e = 0.1 # chance of taking a random action
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  i = 0

  while i < total_episodes:
    if np.random.rand(1) < e:
      action = np.random.randint(len_slot_arms)
    else:
      action = sess.run(chosen_action)
    reward = findReward(slot_arms[action])
    _,resp,ww = sess.run([update, responsible_weight, weights], feed_dict={reward_holder: [reward], action_holder: [action]})
    total_reward[action] += reward
    if i % 50 == 0:
      print ("Running reward for the n=6 arms of slot machine: " + str(total_reward))
    i+=1

print ("The agent thinks bandit " + str(np.argmax(ww)+1) + " has highest probability of giving positive reward")
if np.argmax(ww) == np.argmax(-np.array(slot_arms)):
  print("which is right.")
else:
  print("which is wrong.")
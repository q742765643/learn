#!/usr/bin/python
import paddle

def mlp(input_x, input_y, hid_dim=128, label_dim=2):
  fc_1 = paddle.static.nn.fc(input=input_x, size=hid_dim, act='tanh')
  fc_2 = paddle.static.nn.fc(input=fc_1, size=hid_dim, act='tanh')
  prediction = paddle.static.nn.fc(input=[fc_2], size=label_dim, act='softmax')
  cost = paddle.static.nn.cross_entropy(input=prediction, label=input_y)
  avg_cost = paddle.static.nn.mean(x=cost)
  return avg_cost

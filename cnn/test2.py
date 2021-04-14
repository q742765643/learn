import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker
import numpy as np
import os;
os.environ["PADDLE_PSERVER_NUMS"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"

os.environ["POD_IP"] = "127.0.0.1"
os.environ["PADDLE_PORT"] = "36001"
os.environ["TRAINING_ROLE"] = "TRAINER"
os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
    "127.0.0.1:36001"

os.environ["PADDLE_TRAINER_ID"] = "0"
input_x = fluid.data(name="x", shape=[None, 32], dtype='float32')
input_y = fluid.data(name="y", shape=[None, 1], dtype='int64')

def mlp(input_x, input_y, hid_dim=128, label_dim=2):
  fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
  fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
  prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
  cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
  avg_cost = fluid.layers.mean(x=cost)
  return avg_cost
def gen_data():
    return {"x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64')}
cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(cost)

if fleet.is_server():
  fleet.init_server()
  fleet.run_server()
elif fleet.is_worker():
  fleet.init_worker();
  place = fluid.CPUPlace()
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())
  step = 1001
  for i in range(step):
    cost_val = exe.run(
        program=fluid.default_main_program(),
        feed=gen_data(),
        fetch_list=[cost.name])
    print("worker_index: %d, step%d cost = %f" %
         (fleet.worker_index(), i, cost_val[0]))
  fleet.is_first_worker()

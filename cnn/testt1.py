import paddle
paddle.enable_static()

import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet

from nets import mlp
from utils import gen_data

input_x = paddle.static.data(name="x", shape=[None, 32], dtype='float32')
input_y = paddle.static.data(name="y", shape=[None, 1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = paddle.optimizer.SGD(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)

strategy = paddle.distributed.fleet.DistributedStrategy()
strategy.a_sync = True

optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(cost)

if fleet.is_server():
  fleet.init_server()
  fleet.run_server()

elif fleet.is_worker():
  place = paddle.CPUPlace()
  exe = paddle.static.Executor(place)
  exe.run(paddle.static.default_startup_program())

  step = 1001
  for i in range(step):
    cost_val = exe.run(
        program=paddle.static.default_main_program(),
        feed=gen_data(),
        fetch_list=[cost.name])
    print("worker_index: %d, step%d cost = %f" %
         (fleet.worker_index(), i, cost_val[0]))

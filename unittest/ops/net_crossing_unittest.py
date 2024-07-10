##
# @file  net_crossing.py
# @author Niansong Zhang
# @date  Jul 2024
#

import os
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.net_crossing import net_crossing

sys.path.pop()


def bell_func(x, lambda_, mu_, sigma_):
    if torch.abs(x) <= 0.5:
        return 1 - lambda_ * torch.pow(x, 2)
    elif torch.abs(x) <= 1.5:
        return mu_ * torch.pow(torch.abs(x) - sigma_, 2)
    else:
        return 0


def golden_netcrossing(pin_x, pin_y, pin2net_map, net2pin_map, _lambda, _mu, _sigma):
    num_nets = len(net2pin_map)
    net_crossing = torch.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(num_nets):
        for j in range(i+1, num_nets):
            pins_i_idx = net2pin_map[i]
            pins_j_idx = net2pin_map[j]
            for i_sink_pin_idx in pins_i_idx[1:]:
                x1 = pin_x[pins_i_idx[0]]
                y1 = pin_y[pins_i_idx[0]]
                x2 = pin_x[i_sink_pin_idx]
                y2 = pin_y[i_sink_pin_idx]
                for j_sink_pin_idx in pins_j_idx[1:]:
                    x3 = pin_x[pins_j_idx[0]]
                    y3 = pin_y[pins_j_idx[0]]
                    x4 = pin_x[j_sink_pin_idx]
                    y4 = pin_y[j_sink_pin_idx]
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    )
                    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    )
                    # print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, x3={x3}, y3={y3}, x4={x4}, y4={y4}")
                    # print(f"golden t = {t}, u = {u}")
                    net_crossing[i] += bell_func(
                        t - 0.5, _lambda, _mu, _sigma
                    ) * bell_func(u - 0.5, _lambda, _mu, _sigma)
                    # f = bell_func(t - 0.5, _lambda, _mu, _sigma)
                    # g = bell_func(u - 0.5, _lambda, _mu, _sigma)
    return net_crossing.sum()

class NetCrossingOpTest(unittest.TestCase):
    def test_net_crossing_random(self):
        dtype = torch.float32
        pin_pos = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], dtype=np.float32
        )
        net2pin_map = np.array([np.array([1, 2]), np.array([0, 3, 4])])
        pin2net_map = np.zeros(len(pin_pos), dtype=np.int32)
        for net_id, pins in enumerate(net2pin_map):
            for pin in pins:
                pin2net_map[pin] = net_id
        net_weights = np.array([1, 2], dtype=np.float32)

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]
        ignore_net_degree = 4
        pin_mask = np.zeros(len(pin2net_map), dtype=np.uint8)

        # net mask
        net_mask = np.ones(len(net2pin_map), dtype=np.uint8)
        for i in range(len(net2pin_map)):
            if len(net2pin_map[i]) >= ignore_net_degree:
                net_mask[i] = 0

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        flat_net2pin_start_map = np.zeros(len(net2pin_map) + 1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count : count + len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)

        # print("flat_net2pin_map = ", flat_net2pin_map)
        # print("flat_net2pin_start_map = ", flat_net2pin_start_map)

        # print(np.transpose(pin_pos))
        pin_pos_var = Variable(
            torch.from_numpy(np.transpose(pin_pos)).reshape([-1]), requires_grad=True
        )
        # pin_pos_var = torch.nn.Parameter(torch.from_numpy(np.transpose(pin_pos)).reshape([-1]))
        # print(pin_pos_var)

        _lambda = 2
        _mu = 2
        _sigma = 1

        golden = golden_netcrossing(
            pin_pos_var[: pin_pos_var.numel() // 2],
            pin_pos_var[pin_pos_var.numel() // 2 :],
            pin2net_map,
            net2pin_map,
            _lambda,
            _mu,
            _sigma,
        )
        golden.backward()
        golden_grad = pin_pos_var.grad.clone()

        # print("golden: ", golden)
        # print("golden_grad: ", golden_grad)


        pin_pos_var.grad.zero_()
        custom = net_crossing.NetCrossing(
            flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)),
            netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)),
            net_mask=torch.from_numpy(net_mask),
            _lambda=torch.tensor(_lambda, dtype=dtype),
            _mu=torch.tensor(_mu, dtype=dtype),
            _sigma=torch.tensor(_sigma, dtype=dtype),
        )
        result = custom.forward(pin_pos_var)
        result.backward()
        grad = pin_pos_var.grad.clone()

        # print("custom_result = ", result)
        # print("custom_grad = ", grad)

        np.testing.assert_allclose(result.data.numpy(), golden.data.detach().numpy(), atol=1e-5)
        np.testing.assert_allclose(grad.data.numpy(), golden_grad.data.numpy(), atol=1e-5)

        print("\033[92mCPU test passed!\033[0m")

        # test gpu
        if torch.cuda.device_count():
            pin_pos_var.grad.zero_()
            custom_cuda = net_crossing.NetCrossing(
                flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
                netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                _lambda=torch.tensor(_lambda, dtype=dtype).cuda(),
                _mu=torch.tensor(_mu, dtype=dtype).cuda(),
                _sigma=torch.tensor(_sigma, dtype=dtype).cuda(),
            )
            result_cuda = custom_cuda.forward(pin_pos_var.cuda())
            # print("custom_cuda_result = ", result_cuda.data.cpu())
            result_cuda.backward()
            grad_cuda = pin_pos_var.grad.clone()
            # print("custom_cuda_grad = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(result_cuda.data.cpu().numpy(), golden.data.detach().numpy(), atol=1e-5)
            np.testing.assert_allclose(grad_cuda.data.cpu().numpy(), grad.data.numpy(), atol=1e-5)
            print("\033[92mGPU test passed!\033[0m")

if __name__ == '__main__':
    unittest.main()

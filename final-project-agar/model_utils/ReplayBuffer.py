import random


class ReplayBuffer():
    def __init__(self, capacity, prefill_amt=1):
        if capacity is None or capacity <= 0:
            raise ValueError('Capacity must be a positive integer')

        self.capacity = capacity
        self.prefill_amt = prefill_amt
        self.buffer = []
        self.idx = 0

    def push(self, memory):
        """Save a state transition memory"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.idx] = memory
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        """Get a random sample of [batch_size] memories"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the number of state examples in the buffer"""
        return len(self.buffer)

    def prefill_capacity(self):
        return len(self.buffer) / self.prefill_amt

    def equals(self, other):
        eq_capacities = self.capacity == other.capacity
        eq_prefill_amts = self.prefill_amt == other.prefill_amt
        eq_idxs = self.idx == other.idx

        eq_bufs = True
        for (
            (self_s0, self_act, self_s1, self_reward, self_done),
            (other_s0, other_act, other_s1, other_reward, other_done)
        ) in zip(self.buffer, other.buffer):
            eq_s0s = (self_s0 == other_s0).all()
            eq_act = self_act == other_act
            eq_s1s = (self_s1 == other_s1).all()
            eq_rewards = self_reward == other_reward
            eq_done = self_done == other_done

            eq_bufs = eq_bufs and eq_s0s and eq_act and eq_s1s and eq_rewards and eq_done

        return eq_capacities and eq_prefill_amts and eq_idxs and eq_bufs

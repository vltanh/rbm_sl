import torch
import matplotlib.pyplot as plt
import networkx as nx

DEBUG = False
CUDA = False
BRUTE_FORCE = True


class TreeRBMFactorGraph:
    def __init__(self, W, y) -> None:
        '''
        Initialize a factor graph of a tree-RBM

        Let
            n_v: number of visible nodes
            n_h: number of hidden nodes
            n: n_v + n_h
            B: batch size

        Args:
            W: [n_v, n_h]
            y: [B, n]
        '''
        self.W, self.y = W, y
        self.values = torch.FloatTensor([1.0, -1.0])
        self.reset()

        self.n_v, self.n_h = W.shape
        self.B, self.n = y.shape
        assert self.n == self.n_v + self.n_h

    def reset(self) -> None:
        # (2 * #edges * #directions + #nodes) * #values
        self.messages = {
            'fv2v': dict(),
            'fh2h': dict(),
            'f2v': dict(),
            'f2h': dict(),
            'h2f': dict(),
            'v2f': dict(),
        }

    @staticmethod
    def visualize(W: torch.Tensor, y: torch.Tensor) -> None:
        '''
        Visualize the factor graph of a tree-RBM

        Let
            n_v: number of visible nodes
            n_h: number of hidden nodes
            n: n_v + n_h

        Args:
            W (torch.Tensor): [n_v, n_h]
            y (torch.Tensor): [n]

        Returns:
            None
        '''
        n_v, n_h = W.shape
        n = y.shape
        assert n == n_v + n_h

        G = nx.Graph()

        for i in range(n_v):
            G.add_node(f'$v_{i}$')
            if y[i] != 0.0:
                G.add_node(f'$f^v_{i}$')
                G.add_edge(f'$v_{i}$', f'$f^v_{i}$', weight=y[i])

        for i in range(n_h):
            G.add_node(f'$h_{i}$')
            if y[i + n_v] != 0.0:
                G.add_node(f'$f^h_{i}$')
                G.add_edge(f'$h_{i}$', f'$f^h_{i}$', weight=y[i + n_v])

        for i in range(n_v):
            for j in range(n_h):
                if W[i, j] != 0:
                    G.add_node(f'$f_{{{i},{j}}}$')
                    G.add_edge(f'$v_{i}$', f'$f_{{{i},{j}}}$', weight=W[i, j])
                    G.add_edge(f'$h_{j}$', f'$f_{{{i},{j}}}$', weight=W[i, j])

        nx.draw_planar(G, with_labels=True)
        plt.show()

    def barycenter(self) -> torch.Tensor:
        '''
        Compute the barycenter of a tree-RBM using factor graph and sum-product

        Let
            n_v: number of visible nodes
            n_h: number of hidden nodes
            n: n_v + n_h
            B: batch size

        Returns:
            barycenter (torch.Tensor): [B, n]
        '''
        marginals = self.marginals()  # [B, n]
        barycenter = 2 * marginals - 1  # [B, n]
        return barycenter.clone()  # [B, n]

    def marginals(self) -> torch.Tensor:
        '''
        Compute the marginals of a tree-RBM using factor graph and sum-product

        Let
            n_v: number of visible nodes
            n_h: number of hidden nodes
            n: n_v + n_h
            B: batch size

        Returns:
            m: [B, n]
        '''
        if CUDA:
            W = W.cuda()
            y = y.cuda()
            self.values = self.values.cuda()

        # Store the marginals
        m = torch.zeros(self.B, self.values.size(0), self.n)  # [B, 2, n]
        
        for i in range(self.n_v):
            m[:, :, i] = \
                self.compute_logmarginal_v(i)  # [B, 2] -> [B, 2, n]

        for j in range(self.n_v, self.n):
            m[:, :, j] = \
                self.compute_logmarginal_h(j - self.n_v)  # [B, 2] -> [B, 2, n]

        log_partition = torch.logsumexp(m, dim=1, keepdim=True)  # [B, 1, n]
        log_marginals = m - log_partition  # [B, 2, n] - [B, 1, n] = [B, 2, n]
        m_ = torch.exp(log_marginals)  # [B, 2, n] -> [B, 2, n]
        return m_[:, 0].clone()  # [B, 2, n] -> [B, n]

    def compute_logmarginal_v(self, i_v: int) -> torch.Tensor:
        '''
        Compute the log-marginal of v_i

        Args:
            i_v (int): index of the visible node

        Returns:
            m (torch.Tensor): [B, 2] log-marginal
        '''
        if DEBUG:
            _id = f'v_{i_v}'
            print(f'[COMPUTE] {_id}')

        m = self.compute_message_fv2v(i_v)  # [B, 2]

        for j in range(self.n_h):
            if self.W[i_v, j] != 0:
                # [B, 2] + [B, 2] = [B, 2]
                m += self.compute_message_f2v(i_v, j)

        return m.clone()  # [B, 2]

    def compute_logmarginal_h(self, i_h: int) -> torch.Tensor:
        '''
        Compute the log-marginal of h_i

        Args:
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, n] log-marginal
        '''
        if DEBUG:
            _id = f'h_{i_h}'
            print(f'[COMPUTE] {_id}')

        m = self.compute_message_fh2h(i_h)  # [B, 2]

        for i in range(self.n_v):
            if self.W[i, i_h] != 0:
                # [B, 2] + [B, 2] = [B, 2]
                m += self.compute_message_f2h(i, i_h)

        return m.clone()  # [B, 2]

    def compute_message_fv2v(self, i_v: int) -> torch.Tensor:
        '''
        Compute the message sent from f^v_i to v_i

        Args:
            i_v (int): index of the visible node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f^v_{i_v} -> v_{i_v}'

        if i_v not in self.messages['fv2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            # [B, 1] * [1, 2] = [B, 2]
            m = self.y[:, [i_v]] * self.values[None]

            self.messages['fv2v'][i_v] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return self.messages['fv2v'][i_v].clone()  # [B, 2]

    def compute_message_fh2h(self, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from f^h_i to h_i

        Args:
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f^h_{i_h} -> h_{i_h}'

        if i_h not in self.messages['fh2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            # [B, 1] * [1, 2] = [B, 2]
            m = self.y[:, [self.n_v + i_h]] * self.values[None]

            self.messages['fh2h'][i_h] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return self.messages['fh2h'][i_h].clone()  # [B, 2]

    def compute_message_f2v(self, i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from f_{i_v, i_h} to v_i

        Args:
            i_v (int): index of the visible node
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> v_{i_v}'

        if (i_v, i_h) not in self.messages['f2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            # [1, 1] * [2, 1] * [1, 2] = [2, 2]
            vWh = self.W[i_v, i_h] * \
                self.values[:, None] * self.values[None, :]
            h2f = self.compute_message_h2f(i_v, i_h)  # [B, 2]
            # [2, 1, 2] + [1, B, 2] = [2, B, 2]
            vWh_h2f = vWh[:, None, :] + h2f[None, :, :]

            # [2, B, 2] -> [2, B, 1]
            notsum = torch.logsumexp(vWh_h2f, dim=-1, keepdim=True)
            # [2, B, 1] -> [B, 2]
            m = notsum.squeeze(-1).transpose(0, 1)

            self.messages['f2v'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return self.messages['f2v'][(i_v, i_h)].clone()  # [B, 2]

    def compute_message_f2h(self, i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from f_{i_v, i_h} to h_i

        Args:
            i_v (int): index of the visible node
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> h_{i_h}'

        if (i_v, i_h) not in self.messages['f2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            # [1, 1] * [2, 1] * [1, 2] = [2, 2]
            vWh = self.W[i_v, i_h] * \
                self.values[:, None] * self.values[None, :]
            # [2, 1, 2] + [1, B, 2] = [2, B, 2]
            v2f = self.compute_message_v2f(i_v, i_h)
            # [2, B, 2] -> [2, B, 1]
            vWh_v2f = vWh[:, None, :] + v2f[None, :, :]

            # [2, B, 2] -> [2, B, 1]
            notsum = torch.logsumexp(vWh_v2f, dim=-1, keepdim=True)
            # [2, B, 1] -> [B, 2]
            m = notsum.squeeze(-1).transpose(0, 1)

            self.messages['f2h'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return self.messages['f2h'][(i_v, i_h)].clone()  # [B, 2]

    def compute_message_h2f(self, i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from h_i to f_{i_v, i_h}

        Args:
            i_v (int): index of the visible node
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'h_{i_h} -> f_{{{i_v},{i_h}}}'

        if (i_v, i_h) not in self.messages['h2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = self.compute_message_fh2h(i_h)  # [B, 2]
            for i in range(self.n_v):
                if i != i_v and self.W[i, i_h] != 0:
                    # [B, 2] + [B, 2] = [B, 2]
                    m += self.compute_message_f2h(i, i_h).clone()

            self.messages['h2f'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return self.messages['h2f'][(i_v, i_h)].clone()  # [B, 2]

    def compute_message_v2f(self, i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from v_i to f_{i_v, i_h}

        Args:
            i_v (int): index of the visible node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'v_{i_v} -> f_{{{i_v},{i_h}}}'

        if (i_v, i_h) not in self.messages['v2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = self.compute_message_fv2v(i_v)  # [B, 2]
            for j in range(self.n_h):
                if j != i_h and self.W[i_v, j] != 0:
                    m += self.compute_message_f2v(i_v, j).clone()  # [B, 2]

            self.messages['v2f'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return self.messages['v2f'][(i_v, i_h)].clone()  # [B, 2]

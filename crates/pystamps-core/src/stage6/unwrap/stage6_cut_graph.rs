use std::collections::VecDeque;

const INF_CAP: i64 = 1_i64 << 58;

#[derive(Clone, Copy)]
struct Arc {
    to: usize,
    rev: usize,
    cap: i64,
}

pub(crate) struct Dinic {
    graph: Vec<Vec<Arc>>,
}

impl Dinic {
    pub(crate) fn new(nodes: usize) -> Self {
        Self {
            graph: vec![Vec::new(); nodes],
        }
    }

    pub(crate) fn add_arc(&mut self, from: usize, to: usize, cap: i64) {
        if cap <= 0 {
            return;
        }
        let rev_to = self.graph[to].len();
        let rev_from = self.graph[from].len();
        self.graph[from].push(Arc {
            to,
            rev: rev_to,
            cap,
        });
        self.graph[to].push(Arc {
            to: from,
            rev: rev_from,
            cap: 0,
        });
    }

    pub(crate) fn add_cut_edge(&mut self, a: usize, b: usize, cap: i64) {
        self.add_arc(a, b, cap);
        self.add_arc(b, a, cap);
    }

    fn bfs(&self, source: usize, sink: usize, level: &mut [i32]) -> bool {
        level.fill(-1);
        let mut queue = VecDeque::new();
        level[source] = 0;
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            for edge in &self.graph[node] {
                if edge.cap > 0 && level[edge.to] < 0 {
                    level[edge.to] = level[node] + 1;
                    queue.push_back(edge.to);
                }
            }
        }
        level[sink] >= 0
    }

    fn dfs(
        &mut self,
        node: usize,
        sink: usize,
        flow: i64,
        level: &[i32],
        iter: &mut [usize],
    ) -> i64 {
        if node == sink {
            return flow;
        }
        while iter[node] < self.graph[node].len() {
            let edge_ix = iter[node];
            let edge = self.graph[node][edge_ix];
            if edge.cap > 0 && level[node] + 1 == level[edge.to] {
                let pushed = self.dfs(edge.to, sink, flow.min(edge.cap), level, iter);
                if pushed > 0 {
                    self.graph[node][edge_ix].cap -= pushed;
                    self.graph[edge.to][edge.rev].cap += pushed;
                    return pushed;
                }
            }
            iter[node] += 1;
        }
        0
    }

    pub(crate) fn max_flow(&mut self, source: usize, sink: usize) {
        let mut level = vec![-1; self.graph.len()];
        while self.bfs(source, sink, &mut level) {
            let mut iter = vec![0; self.graph.len()];
            loop {
                let pushed = self.dfs(source, sink, INF_CAP, &level, &mut iter);
                if pushed == 0 {
                    break;
                }
            }
        }
    }

    pub(crate) fn reachable_from(&self, source: usize) -> Vec<bool> {
        let mut seen = vec![false; self.graph.len()];
        let mut queue = VecDeque::new();
        seen[source] = true;
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            for edge in &self.graph[node] {
                if edge.cap > 0 && !seen[edge.to] {
                    seen[edge.to] = true;
                    queue.push_back(edge.to);
                }
            }
        }
        seen
    }
}

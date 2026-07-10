pub(crate) struct CompactDisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl CompactDisjointSet {
    pub(crate) fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            rank: vec![0; count],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    pub(crate) fn union(&mut self, left: usize, right: usize) -> bool {
        let mut root_left = self.find(left);
        let mut root_right = self.find(right);
        if root_left == root_right {
            return false;
        }
        if self.rank[root_left] < self.rank[root_right] {
            std::mem::swap(&mut root_left, &mut root_right);
        }
        self.parent[root_right] = root_left;
        if self.rank[root_left] == self.rank[root_right] {
            self.rank[root_left] += 1;
        }
        true
    }
}

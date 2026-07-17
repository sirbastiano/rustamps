use std::cmp::Ordering;

use rayon::prelude::*;

pub fn nearest_grid(points: &[[f64; 2]], rows: usize, cols: usize) -> Result<Vec<usize>, String> {
    let indexed = points
        .iter()
        .enumerate()
        .map(|(id, point)| Point { id, xy: *point })
        .collect();
    let tree =
        Node::build(indexed, 0).ok_or_else(|| "cannot interpolate an empty grid".to_owned())?;
    Ok((0..rows * cols)
        .into_par_iter()
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            tree.nearest(col as f64, row as f64).0
        })
        .collect())
}

#[derive(Clone, Copy)]
struct Point {
    id: usize,
    xy: [f64; 2],
}

struct Node {
    point: Point,
    axis: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    fn build(mut points: Vec<Point>, depth: usize) -> Option<Box<Self>> {
        if points.is_empty() {
            return None;
        }
        let axis = depth % 2;
        points.sort_by(|a, b| a.xy[axis].total_cmp(&b.xy[axis]).then(a.id.cmp(&b.id)));
        let right = points.split_off(points.len() / 2 + 1);
        let point = points.pop()?;
        Some(Box::new(Self {
            point,
            axis,
            left: Self::build(points, depth + 1),
            right: Self::build(right, depth + 1),
        }))
    }

    fn nearest(&self, x: f64, y: f64) -> (usize, f64) {
        let mut best = (self.point.id, distance(self.point, x, y));
        self.search(x, y, &mut best);
        best
    }

    fn search(&self, x: f64, y: f64, best: &mut (usize, f64)) {
        let distance = distance(self.point, x, y);
        if distance < best.1 || (distance == best.1 && self.point.id < best.0) {
            *best = (self.point.id, distance);
        }
        let query = if self.axis == 0 { x } else { y };
        let delta = query - self.point.xy[self.axis];
        let (near, far) = if delta.total_cmp(&0.0) == Ordering::Less {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };
        if let Some(node) = near {
            node.search(x, y, best);
        }
        if delta * delta <= best.1 {
            if let Some(node) = far {
                node.search(x, y, best);
            }
        }
    }
}

fn distance(point: Point, x: f64, y: f64) -> f64 {
    (point.xy[0] - x).mul_add(point.xy[0] - x, (point.xy[1] - y) * (point.xy[1] - y))
}

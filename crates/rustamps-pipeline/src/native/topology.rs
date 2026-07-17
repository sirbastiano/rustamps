use std::collections::BTreeSet;

use delaunator::{triangulate, Point};

pub fn delaunay_edges(xy: &[f64]) -> Result<Vec<[usize; 2]>, String> {
    if xy.len() % 2 != 0 || xy.iter().any(|value| !value.is_finite()) {
        return Err("Delaunay coordinates must be finite x/y pairs".to_owned());
    }
    let points = xy
        .chunks_exact(2)
        .map(|pair| Point {
            x: pair[0],
            y: pair[1],
        })
        .collect::<Vec<_>>();
    if points.len() < 3 {
        return Ok(Vec::new());
    }
    let triangulation = triangulate(&points);
    let mut edges = BTreeSet::new();
    for triangle in triangulation.triangles.chunks_exact(3) {
        insert(&mut edges, triangle[0], triangle[1]);
        insert(&mut edges, triangle[1], triangle[2]);
        insert(&mut edges, triangle[2], triangle[0]);
    }
    Ok(edges
        .into_iter()
        .map(|(left, right)| [left, right])
        .collect())
}

fn insert(edges: &mut BTreeSet<(usize, usize)>, left: usize, right: usize) {
    if left != right {
        edges.insert((left.min(right), left.max(right)));
    }
}

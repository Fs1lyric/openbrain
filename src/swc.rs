//! Parse SWC morphology files (as served by NeuroMorpho.Org) into cable geometry.
//!
//! The root point becomes a soma cylinder whose length equals its diameter (same membrane area as a
//! sphere). Every other point becomes a cylinder running from its parent point to itself.

use std::collections::HashMap;

use crate::cable::Geometry;

struct Point {
    xyz: [f64; 3],
}

pub fn parse(text: &str) -> Result<Geometry, String> {
    let mut index: HashMap<i64, usize> = HashMap::new();
    let mut points: Vec<Point> = Vec::new();
    let mut geom = Geometry {
        parent: vec![],
        length_um: vec![],
        diameter_um: vec![],
    };
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<f64> = line
            .split_whitespace()
            .take(7)
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|e| format!("line {}: {e}", line_no + 1))?;
        let [id, _kind, x, y, z, radius, parent] = fields[..] else {
            return Err(format!("line {}: expected 7 fields", line_no + 1));
        };
        let point = Point { xyz: [x, y, z] };
        let parent_idx = if parent < 0.0 {
            if !points.is_empty() {
                return Err(format!("line {}: more than one root", line_no + 1));
            }
            None
        } else {
            Some(*index.get(&(parent as i64)).ok_or_else(|| {
                format!("line {}: parent {parent} not defined earlier", line_no + 1)
            })?)
        };
        let length = match parent_idx {
            None => 2.0 * radius,
            Some(p) => {
                let d: f64 = (0..3)
                    .map(|k| (point.xyz[k] - points[p].xyz[k]).powi(2))
                    .sum();
                d.sqrt().max(0.1)
            }
        };
        index.insert(id as i64, points.len());
        geom.parent.push(parent_idx);
        geom.length_um.push(length);
        geom.diameter_um.push(2.0 * radius);
        points.push(point);
    }
    if points.is_empty() {
        return Err("no points".into());
    }
    Ok(geom)
}

use super::super::params::Params;
use super::input::Loaded;

pub fn select(input: &Loaded, params: &Params) -> Result<Vec<usize>, String> {
    validate(params)?;
    let lon_bounds = bounds(params, "ref_lon")?;
    let lat_bounds = bounds(params, "ref_lat")?;
    let radius = params.scalar("ref_radius", f64::INFINITY)?;
    if radius == f64::NEG_INFINITY {
        return Ok(Vec::new());
    }
    let mut selected = (0..input.n_ps)
        .filter(|&row| {
            let lon = input.lonlat[row * 2];
            let lat = input.lonlat[row * 2 + 1];
            lon > lon_bounds[0] && lon < lon_bounds[1] && lat > lat_bounds[0] && lat < lat_bounds[1]
        })
        .collect::<Vec<_>>();
    if radius.is_finite() && !selected.is_empty() {
        let center = params.vector("ref_centre_lonlat", &[0.0, 0.0])?;
        if input.ll0.len() < 2 || input.ll0[..2].iter().any(|value| !value.is_finite()) {
            return Err("ps2.ll0 must contain a finite lon/lat origin".to_owned());
        }
        let origin = [input.ll0[0], input.ll0[1]];
        let center_xy = local_xy(center[0], center[1], origin);
        selected.retain(|&row| {
            let xy = local_xy(input.lonlat[row * 2], input.lonlat[row * 2 + 1], origin);
            let dx = xy[0] - center_xy[0];
            let dy = xy[1] - center_xy[1];
            dx * dx + dy * dy <= radius * radius
        });
    }
    if selected.is_empty() {
        return Err("reference constraints select no PS".to_owned());
    }
    Ok(selected)
}

pub fn validate(params: &Params) -> Result<(), String> {
    if ["ref_x", "ref_y"].iter().any(|key| params.contains(key)) {
        return Err(
            "ref_x/ref_y Cartesian reference bounds are unsupported by native Stage 7".to_owned(),
        );
    }
    bounds(params, "ref_lon")?;
    bounds(params, "ref_lat")?;
    let radius = params.scalar("ref_radius", f64::INFINITY)?;
    if radius.is_nan() || (radius < 0.0 && radius != f64::NEG_INFINITY) {
        return Err("ref_radius must be non-negative, +Inf, or -Inf".to_owned());
    }
    if radius.is_finite() {
        let center = params.vector("ref_centre_lonlat", &[0.0, 0.0])?;
        if center.len() != 2 || center.iter().any(|value| !value.is_finite()) {
            return Err("ref_centre_lonlat must contain exactly two finite values".to_owned());
        }
    }
    Ok(())
}

// WGS84 projection used by StaMPS llh2local.m.
fn local_xy(longitude: f64, latitude: f64, origin: [f64; 2]) -> [f64; 2] {
    let a = 6_378_137.0;
    let eccentricity: f64 = 0.082_094_437_949_70;
    let e2 = eccentricity.powi(2);
    let e4 = eccentricity.powi(4);
    let e6 = eccentricity.powi(6);
    let lat = latitude.to_radians();
    let origin_lat = origin[1].to_radians();
    let delta_lon = longitude.to_radians() - origin[0].to_radians();
    let meridian = |value: f64| {
        a * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * value
            - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * value).sin()
            + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * value).sin()
            - 35.0 * e6 / 3072.0 * (6.0 * value).sin())
    };
    let origin_meridian = meridian(origin_lat);
    if lat == 0.0 {
        return [a * delta_lon, -origin_meridian];
    }
    let prime_vertical = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    let east = delta_lon * lat.sin();
    let cotangent = 1.0 / lat.tan();
    [
        prime_vertical * cotangent * east.sin(),
        meridian(lat) - origin_meridian + prime_vertical * cotangent * (1.0 - east.cos()),
    ]
}

fn bounds(params: &Params, key: &str) -> Result<[f64; 2], String> {
    let values = params.vector(key, &[f64::NEG_INFINITY, f64::INFINITY])?;
    if values.len() != 2 || values[0].is_nan() || values[1].is_nan() || values[0] >= values[1] {
        return Err(format!("{key} must contain exactly two ordered bounds"));
    }
    let default_unbounded = values == [f64::NEG_INFINITY, f64::INFINITY];
    if !default_unbounded && values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{key} bounds must be finite or the StaMPS [-Inf, Inf] default"
        ));
    }
    Ok([values[0], values[1]])
}

#[cfg(test)]
mod tests {
    use super::local_xy;

    #[test]
    fn local_projection_matches_stamps_llh2local() {
        let xy = local_xy(12.501, 41.901, [12.5, 41.9]);
        assert!((xy[0] - 82.979_894_71).abs() < 1.0e-6);
        assert!((xy[1] - 111.070_153_07).abs() < 1.0e-6);
    }
}

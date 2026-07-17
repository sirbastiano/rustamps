use super::types::Stage1Error;

const WGS84_A: f64 = 6_378_137.0;
const WGS84_E: f64 = 0.082_094_437_949_70;

fn meridian_arc(latitude: f64) -> f64 {
    let e = WGS84_E;
    WGS84_A
        * ((1.0 - e.powi(2) / 4.0 - 3.0 * e.powi(4) / 64.0 - 5.0 * e.powi(6) / 256.0) * latitude
            - (3.0 * e.powi(2) / 8.0 + 3.0 * e.powi(4) / 32.0 + 45.0 * e.powi(6) / 1024.0)
                * (2.0 * latitude).sin()
            + (15.0 * e.powi(4) / 256.0 + 45.0 * e.powi(6) / 1024.0) * (4.0 * latitude).sin()
            - 35.0 * e.powi(6) / 3072.0 * (6.0 * latitude).sin())
}

fn extent(values: &[[f64; 2]], component: usize) -> f64 {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for value in values {
        low = low.min(value[component]);
        high = high.max(value[component]);
    }
    high - low
}

pub fn local_xy(
    lonlat: &[[f64; 2]],
    heading_deg: Option<f64>,
) -> Result<(Vec<[f64; 2]>, [f64; 2]), Stage1Error> {
    if lonlat.is_empty() {
        return Err(Stage1Error::NoCandidates);
    }
    let mut lon_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lat_max = f64::NEG_INFINITY;
    for &[lon, lat] in lonlat {
        lon_min = lon_min.min(lon);
        lon_max = lon_max.max(lon);
        lat_min = lat_min.min(lat);
        lat_max = lat_max.max(lat);
    }
    let origin = [(lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0];
    let origin_rad = [origin[0].to_radians(), origin[1].to_radians()];
    let m0 = meridian_arc(origin_rad[1]);
    let mut xy = Vec::with_capacity(lonlat.len());
    for &[lon, lat] in lonlat {
        let longitude = lon.to_radians();
        let latitude = lat.to_radians();
        let delta = longitude - origin_rad[0];
        let point = if latitude != 0.0 {
            let n = WGS84_A / (1.0 - WGS84_E.powi(2) * latitude.sin().powi(2)).sqrt();
            let e_term = delta * latitude.sin();
            let cot = 1.0 / latitude.tan();
            [
                n * cot * e_term.sin(),
                meridian_arc(latitude) - m0 + n * cot * (1.0 - e_term.cos()),
            ]
        } else {
            [WGS84_A * delta, -m0]
        };
        xy.push(point);
    }

    if let Some(heading) = heading_deg {
        let mut theta = (180.0 - heading).to_radians();
        if theta > std::f64::consts::PI {
            theta -= 2.0 * std::f64::consts::PI;
        }
        let rotated = xy
            .iter()
            .map(|&[x, y]| {
                [
                    theta.cos() * x + theta.sin() * y,
                    -theta.sin() * x + theta.cos() * y,
                ]
            })
            .collect::<Vec<_>>();
        if extent(&rotated, 0) < extent(&xy, 0) && extent(&rotated, 1) < extent(&xy, 1) {
            xy = rotated;
        }
    }
    Ok((xy, origin))
}

pub fn quantize_millimeters(value: f64) -> f32 {
    (value as f32 * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_scene_midpoint_and_center_is_near_zero() {
        let points = [[12.0, 45.0], [12.2, 45.2]];
        let (xy, origin) = local_xy(&points, None).unwrap();
        assert_eq!(origin, [12.1, 45.1]);
        assert!(xy[0][0] < 0.0 && xy[1][0] > 0.0);
        assert_eq!(quantize_millimeters(1.2345), 1.235);
    }
}

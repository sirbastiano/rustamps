use super::input::Input;

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;
const MATLAB_INTEGER_MASK: u64 = (1_u64 << 52) - 1;

pub fn input(input: &Input) -> u64 {
    let mut hash = OFFSET;
    integer(&mut hash, input.n_ps as u64);
    integer(&mut hash, input.n_ifg as u64);
    integer(&mut hash, input.master as u64);
    for &index in &input.unwrap {
        integer(&mut hash, index as u64);
    }
    for value in &input.phase {
        integer(&mut hash, u64::from(value.re.to_bits()));
        integer(&mut hash, u64::from(value.im.to_bits()));
    }
    for &value in &input.phase_restore {
        integer(&mut hash, u64::from(value.to_bits()));
    }
    for point in &input.xy {
        integer(&mut hash, point[0].to_bits());
        integer(&mut hash, point[1].to_bits());
    }
    for &value in input.day.iter().chain(&input.bperp) {
        integer(&mut hash, value.to_bits());
    }
    integer(&mut hash, input.options.grid_size.to_bits());
    integer(&mut hash, input.options.prefilter as u64);
    integer(&mut hash, input.options.filter_window as u64);
    integer(&mut hash, input.options.filter_alpha.to_bits());
    integer(&mut hash, input.options.time_window.to_bits());
    integer(&mut hash, input.options.trial_wraps.to_bits());
    integer(
        &mut hash,
        input.options.max_flow_passes.unwrap_or_default() as u64,
    );
    let compact = hash & MATLAB_INTEGER_MASK;
    compact.max(1)
}

fn integer(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(PRIME);
    }
}

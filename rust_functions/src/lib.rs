use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use numpy::{Complex32, Complex64, PyArray1, PyArrayMethods, PyArray};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use pyo3::Bound; 
use pyo3::wrap_pyfunction;

// implements the jitter & drift transform
#[pyfunction]
#[pyo3(signature = (h, x, uprate, drate, jitter_ppm, drift_ppm, seed))]
fn sampling_clock_impairments<'py>(
    py: Python<'py>, 
    h: PyReadonlyArray1<'py,f32>, 
    x: PyReadonlyArray1<'py,Complex32>, 
    uprate: i32, 
    drate: f32, 
    jitter_ppm: f32, 
    drift_ppm: f32, 
    seed: u64
) -> Bound<'py,PyArray1<Complex32>> {

   // run the transform
   let output_vec = irrational_rate_resampler(h.to_vec().unwrap(),&x.to_vec().unwrap(),uprate as u32, drate, jitter_ppm, drift_ppm, seed);
   // return the output
   PyArray::from_vec_bound(py, output_vec)
}

// function to emulate SciPy's upfirdn() call
#[pyfunction]
#[pyo3(signature = (h, x, uprate, drate))]
fn upfirdn<'py>(
    py: Python<'py>, 
    h: PyReadonlyArray1<'py,f32>, 
    x: PyReadonlyArray1<'py,Complex32>, 
    uprate: i32, 
    drate: f32
) -> Bound<'py,PyArray1<Complex32>> {

    // // run the resampler
    // Convert inputs into owned Rust types
    let h_vec = h.to_vec().unwrap();
    let x_vec = x.to_vec().unwrap();

    // Call your Rust resampler function
    // we use nominal seed = 0
    let output_array: Vec<Complex32> = irrational_rate_resampler(h_vec.to_vec(), &x_vec, uprate as u32, drate, 0.0, 0.0, 0);

    // Convert the output Vec<Complex32> into a NumPy array and return
    PyArray::from_vec_bound(py, output_array)
}

// 
// implements an irrational rate resampler using pfb
pub fn irrational_rate_resampler(
    h: Vec<f32>,
    input_samples: &Vec<Complex32>,
    up_rate: u32,
    down_rate: f32,
    jitter_ppm: f32,
    drift_ppm: f32,
    seed: u64
) -> Vec<Complex32> {

    // compute the number of taps per phase in PFB
    let taps_per_phase = (h.len() as f32 / up_rate as f32).ceil() as u32;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal_jitter = Normal::new(0.0, jitter_ppm*1e-6).unwrap();
    let normal_drift = Normal::new(0.0, drift_ppm * 1e-6).unwrap();

    // let mut rng = rand::rng();
    
    // design and partition the polyphase filter bank
    let h_pfb: Vec<Vec<f32>> = partition_polyphase(h, up_rate, taps_per_phase);

    let padded_len: usize = input_samples.len() + taps_per_phase as usize * 2 - 1;
    // zero-pad the samples to flush the sample buffer on output
    let mut input_samples_padded: Vec<Complex32> = vec![Complex32::ZERO; padded_len];  //input_samples.clone();
    let start = taps_per_phase as usize - 1; // minus 2 is to accurately replicate scipy's upfirdn function
    let end = start + input_samples.len();
    input_samples_padded[start..end].copy_from_slice(&input_samples);

    // commutator
    let mut q_step: f32 = up_rate as f32 / down_rate as f32;

    // calculate number of output samples
    let num_output_samples_estimated = (input_samples_padded.len() as f32 * up_rate as f32 / down_rate as f32).ceil() as u32 + 1;
    // pre-allocate memory for the output samples
    let mut output_samples = vec![Complex32::ZERO;num_output_samples_estimated as usize];

    // initialize indexing for input and output sample streams
    let mut input_idx = 0;
    let mut output_idx = 0;
    let mut clock_drift: f32 = 0.0;

    let idx_stop = input_samples_padded.len() - taps_per_phase as usize;

    // run the resampler. run until all input samples are processed
    while input_idx < idx_stop {
        // run commutator to determine how many input samples correspond to output samples
        while q_step >= up_rate as f32 {
            // update commutator position
            q_step -= up_rate as f32;
            // update index into input time series
            input_idx += 1;
        }
        let delay_slice: &[Complex32] = &input_samples_padded[input_idx..input_idx + taps_per_phase as usize];
        if q_step >= up_rate as f32 {
            break;
        }
        // "phase" or branch selector into filter bank
        let phase = q_step as i32 as usize;
        // get filter weights from PFB
        let h_phase = &h_pfb[phase][..taps_per_phase as usize];
        // get input samples from sample buffer
        // implement the multiply and add using iterators
        let (acc_re, acc_im) = h_phase
            .iter()
            .zip(delay_slice.iter().rev())
            .fold((0.0f32, 0.0f32), |(sum_re, sum_im), (&h_coef, &sample)| {
                (
                    sum_re + h_coef * sample.re,
                    sum_im + h_coef * sample.im,
                )
            });

        // define output sample
        let pfb_out = Complex32::new(acc_re, acc_im);

        // place the output sample at the proper index
        output_samples[output_idx] = pfb_out;
        // increment output index
        output_idx += 1;

        // increment commutator based on resampling rate
        if jitter_ppm != 0.0 || drift_ppm != 0.0 {
            let clock_jitter: f32 = normal_jitter.sample(&mut rng);
            clock_drift += normal_drift.sample(&mut rng);
            q_step += down_rate + clock_jitter + clock_drift;

        } else {
            q_step += down_rate;
        }
    }

    // slices output to proper length. subtract off 1 from output_idx
    // because of most recent +1 does not correspond to an output sample
    // output_samples[0..output_idx-1].to_vec()
    if output_idx == 0 {
        vec![]
    } else {
        output_samples[0..output_idx-1].to_vec()
    }
}


// transforms the prototype filter weights into a polyphase filter bank,
// which is a two-dimensional vector
fn partition_polyphase(h: Vec<f32>, up_rate: u32, taps_per_phase: u32) -> Vec<Vec<f32>> {
    // pre-allocate two-dimensional vector which will hold PFB weights
    let mut h_pfb: Vec<Vec<f32>> = vec![vec![0.0 as f32; taps_per_phase as usize]; up_rate as usize];
    // perform the polyphase partition
    for phase in 0..up_rate {
        let mut tap_idx = phase;
        for idx in 0..taps_per_phase {
            if tap_idx >= h.len() as u32 {
                h_pfb[phase as usize][idx as usize] = 0.0;

            } else {
                h_pfb[phase as usize][idx as usize] = h[tap_idx as usize] * up_rate as f32;
            }
            tap_idx += up_rate as u32;
        }
    }
    h_pfb
}

#[pyfunction]
#[pyo3(signature = (data, 
    alpha_smooth= 1e-4,
    alpha_track= 1e-3,
    alpha_overflow =0.1,
    alpha_acquire = 1e-3,
    ref_level_db = 0.0,
    track_range_db = 1.0,
    low_level_db = -80.0,
    high_level_db = 10.0,

))]
fn digital_agctran<'py>(data:PyReadonlyArray1<'py,Complex64>, py: Python<'py>,
    alpha_smooth:f32, alpha_track:f32, alpha_overflow:f32, alpha_acquire:f32, ref_level_db:f32, 
    track_range_db:f32, low_level_db:f32,  high_level_db:f32)-> Bound<'py,PyArray1<Complex64>> {

    let input_vec = data.to_vec().unwrap();
    let mut output = vec![Complex64::ZERO;input_vec.len() as usize];

    let mut gain_db:f32 = 0.0;
    let mut level_db:f32 = 0.0;

    let mut sample_idx:i32 = 0;
    for sample in input_vec {
        if sample.re == 0.0f64 && sample.im == 0.0f64 {
            level_db = -200.0;
        }else if sample_idx == 0 {
            level_db = (sample.re + sample.im).abs().ln() as f32;
        }else{
            level_db = level_db * alpha_smooth + ((sample.re + sample.im).abs().log2() as f32) * (1.0 - alpha_smooth);
        }
        let output_db = level_db + gain_db;
        let diff_db = ref_level_db - output_db;

        let alpha_adjust;
        if level_db <= low_level_db {
            alpha_adjust = 0.0;
        }else if level_db > high_level_db{
            alpha_adjust = alpha_overflow;
        }else if diff_db.abs() > track_range_db {
            alpha_adjust = alpha_track;
        } else {
            alpha_adjust = alpha_acquire;
        }
            
        gain_db += diff_db * alpha_adjust;

        let gain = gain_db.exp();

        output[sample_idx as usize ] = sample * gain as f64;
        sample_idx += 1; 
    }
    // println!("{:?}", output);
    PyArray::from_vec_bound(py, output)
}

// bind rust functions such that they are callable from python
#[pymodule]
fn rust_functions(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(upfirdn, m)?)?;
    m.add_function(wrap_pyfunction!(sampling_clock_impairments, m)?)?;
    m.add_function(wrap_pyfunction!(digital_agctran, m)?)?;
    Ok(())
}

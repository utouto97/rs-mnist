use std::fs::File;
use std::io::{Read, Write};

pub type Matrix = Vec<Vec<f32>>;

pub fn argmax(x: &Matrix) -> Vec<usize> {
    x.iter()
        .map(|v| {
            let (i, _) = v
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |(maxi, maxv), (i, v)| {
                    if maxv < *v {
                        (i, *v)
                    } else {
                        (maxi, maxv)
                    }
                });
            i
        })
        .collect()
}

pub fn scalar(x: &Matrix, k: f32) -> Matrix {
    x.iter()
        .map(|v| v.iter().map(|e| e * k).collect())
        .collect()
}

pub fn addbias(x: &Matrix, y: &Matrix) -> Matrix {
    x.iter()
        .map(|xs| xs.iter().zip(y[0].iter()).map(|(x, y)| x + y).collect())
        .collect()
}

pub fn matadd(x: &Matrix, y: &Matrix) -> Matrix {
    x.into_iter()
        .zip(y.into_iter())
        .map(|(xs, ys)| {
            xs.into_iter()
                .zip(ys.into_iter())
                .map(|(x, y)| x + y)
                .collect()
        })
        .collect()
}

pub fn matmul(x: &Matrix, y: &Matrix) -> Matrix {
    let r = x.len();
    let k = x[0].len();
    let c = y[0].len();
    let mut result = vec![vec![0.0; c]; r];
    for i in 0..r {
        for j in 0..c {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += x[i][kk] * y[kk][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

pub fn transpose(x: &Matrix) -> Matrix {
    let r = x.len();
    let c = x[0].len();
    let mut result = vec![vec![0.0; r]; c];
    for i in 0..r {
        for j in 0..c {
            result[j][i] = x[i][j];
        }
    }
    result
}

pub fn save_matrix(fname: &str, x: &Matrix) {
    let mut file = File::create(fname).unwrap();
    for row in x {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                row.as_ptr() as *const u8,
                row.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();
    }
    file.flush().unwrap();
}

pub fn load_matrix(fname: &str, x: &mut Matrix) {
    let file = File::open(fname).unwrap();
    let metadata = file.metadata().unwrap();
    let file_size = metadata.len();

    let num_rows = (file_size / (std::mem::size_of::<f32>() as u64)) as usize;

    let mut buffer = vec![0; num_rows * std::mem::size_of::<f32>()];

    let mut file = file;
    file.read_exact(&mut buffer).unwrap();

    unsafe {
        let ptr = buffer.as_ptr() as *const f32;
        let slice = std::slice::from_raw_parts(ptr, num_rows);

        for i in 0..x.len() {
            for j in 0..x[i].len() {
                x[i][j] = slice[i * x[i].len() + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let m2 = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        let m3 = vec![
            vec![11.0, 4.0, 7.0],
            vec![2.0, -5.0, -8.0],
            vec![13.0, -6.0, 9.0],
        ];
        assert_eq!(argmax(&m1), vec![2, 2, 2]);
        assert_eq!(argmax(&m2), vec![2, 2, 2]);
        assert_eq!(argmax(&m3), vec![0, 0, 0]);
    }

    #[test]
    fn test_scalar() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        assert_eq!(
            scalar(&m1, 3.5),
            vec![
                vec![3.5, 7.0, 10.5],
                vec![14.0, 17.5, 21.0],
                vec![24.5, 28.0, 31.5]
            ]
        );
        assert_eq!(
            scalar(&m1, -1.0),
            vec![
                vec![-1.0, -2.0, -3.0],
                vec![-4.0, -5.0, -6.0],
                vec![-7.0, -8.0, -9.0]
            ]
        );
    }

    #[test]
    fn test_addbias() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let m2 = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        let b = vec![vec![1.0, -2.0, 3.0]];
        assert_eq!(
            addbias(&m1, &b),
            vec![
                vec![2.0, 0.0, 6.0],
                vec![5.0, 3.0, 9.0],
                vec![8.0, 6.0, 12.0]
            ]
        );
        assert_eq!(
            addbias(&m2, &b),
            vec![
                vec![2.0, 2.0, 10.0],
                vec![3.0, 3.0, 11.0],
                vec![4.0, 4.0, 12.0]
            ]
        );
    }

    #[test]
    fn test_matadd() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let m2 = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        assert_eq!(
            matadd(&m1, &m2),
            vec![
                vec![2.0, 6.0, 10.0],
                vec![6.0, 10.0, 14.0],
                vec![10.0, 14.0, 18.0]
            ]
        );
        assert_eq!(
            matadd(&scalar(&m1, -1.0), &m2),
            vec![
                vec![0.0, 2.0, 4.0],
                vec![-2.0, 0.0, 2.0],
                vec![-4.0, -2.0, 0.0]
            ]
        );
    }

    #[test]
    fn test_matmul() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let m2 = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        assert_eq!(
            matmul(&m1, &m2),
            vec![
                vec![14.0, 32.0, 50.0],
                vec![32.0, 77.0, 122.0],
                vec![50.0, 122.0, 194.0]
            ]
        );
        assert_eq!(
            matmul(&m2, &m1),
            vec![
                vec![66.0, 78.0, 90.0],
                vec![78.0, 93.0, 108.0],
                vec![90.0, 108.0, 126.0]
            ]
        );
        assert_eq!(
            matmul(&scalar(&m1, -1.0), &m2),
            vec![
                vec![-14.0, -32.0, -50.0],
                vec![-32.0, -77.0, -122.0],
                vec![-50.0, -122.0, -194.0]
            ]
        );
    }

    #[test]
    fn test_transpose() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let m2 = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        assert_eq!(transpose(&m1), m2);
        assert_eq!(transpose(&transpose(&m1)), m1);
    }

    #[test]
    fn test_save_and_load() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mut m2 = vec![vec![0.0; 3]; 3];
        save_matrix("tmp/testdata.bin", &m1);
        load_matrix("tmp/testdata.bin", &mut m2);
        assert_eq!(m1, m2);
    }
}

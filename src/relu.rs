use crate::layer::Layer;
use crate::matrix::Matrix;

pub struct Relu {}

impl Relu {
    pub fn new() -> Self {
        Self {}
    }
}

impl Layer for Relu {
    fn forward(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|v| v.iter().map(|e| if *e > 0.0 { *e } else { 0.0 }).collect())
            .collect()
    }

    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix {
        x.iter()
            .zip(grad_output.iter())
            .map(|(x, g)| {
                x.iter()
                    .zip(g.iter())
                    .map(|(x, g)| if *x > 0.0 { *g } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    fn save(&self) {}

    fn load(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![-7.0, -8.0, -9.0],
        ];
        let m2 = vec![
            vec![1.0, -4.0, 7.0],
            vec![2.0, -5.0, 8.0],
            vec![3.0, -6.0, 9.0],
        ];
        let relu = Relu::new();
        assert_eq!(
            relu.forward(&m1),
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(
            relu.forward(&m2),
            vec![
                vec![1.0, 0.0, 7.0],
                vec![2.0, 0.0, 8.0],
                vec![3.0, 0.0, 9.0],
            ]
        );
    }

    #[test]
    fn test_relu_backward() {
        let m1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![-7.0, -8.0, -9.0],
        ];
        let m2 = vec![
            vec![1.0, -4.0, 7.0],
            vec![2.0, -5.0, 8.0],
            vec![3.0, -6.0, 9.0],
        ];
        let mut relu = Relu::new();
        assert_eq!(
            relu.backward(&m1, &m2),
            vec![
                vec![1.0, -4.0, 7.0],
                vec![2.0, -5.0, 8.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(
            relu.backward(&m2, &m1),
            vec![
                vec![1.0, 0.0, 3.0],
                vec![4.0, 0.0, 6.0],
                vec![-7.0, 0.0, -9.0],
            ]
        );
    }
}

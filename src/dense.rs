use crate::{addbias, load_matrix, matadd, matmul, save_matrix, scalar, transpose, Layer, Matrix};
use rand::prelude::{thread_rng, Distribution};
use rand_distr::Normal;

fn initialize_matrix(x: &mut Matrix) {
    let mut rng = thread_rng();
    let dist = Normal::<f32>::new(0.0, 0.01).unwrap();
    for i in 0..x.len() {
        for j in 0..x[i].len() {
            x[i][j] = dist.sample(&mut rng);
        }
    }
}

pub struct Dense {
    lr: f32,
    weights: Matrix,
    bias: Matrix,
    name: String,
}

impl Dense {
    pub fn new(n_inputs: usize, n_outputs: usize, lr: f32, name: String) -> Self {
        let mut weights = vec![vec![0.0; n_outputs]; n_inputs];
        initialize_matrix(&mut weights);
        let bias = vec![vec![0.0; n_outputs]; 1];
        Self {
            lr,
            weights,
            bias,
            name,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, x: &Matrix) -> Matrix {
        addbias(&matmul(&x, &self.weights), &self.bias)
    }

    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix {
        let grad_input = matmul(&grad_output, &transpose(&self.weights));

        let grad_weights = matmul(&transpose(&x), &grad_output);
        let grad_bias = vec![transpose(&grad_output)
            .iter()
            .map(|v| {
                let sum = v.iter().fold(0.0, |acc, e| acc + e);
                sum
            })
            .collect()];

        self.weights = matadd(&self.weights, &scalar(&grad_weights, -self.lr));
        self.bias = matadd(&self.bias, &scalar(&grad_bias, -self.lr));

        grad_input
    }

    fn save(&self) {
        save_matrix(&format!("params/{}.weights.bin", self.name), &self.weights);
        save_matrix(&format!("params/{}.bias.bin", self.name), &self.bias);
    }

    fn load(&mut self) {
        load_matrix(
            &format!("params/{}.weights.bin", self.name),
            &mut self.weights,
        );
        load_matrix(&format!("params/{}.bias.bin", self.name), &mut self.bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward1() {
        let mut dense = Dense::new(3, 3, 0.0, "0".to_string());
        dense.weights = vec![
            vec![-0.00377593, 0.00922437, -0.00326532],
            vec![-0.00185564, -0.01271753, -0.00150554],
            vec![-0.01585257, -0.00798056, 0.00427046],
        ];
        dense.bias = vec![vec![0.28165264, -0.05307488, 0.3428101]];
        let x = vec![
            vec![-0.43017406, 0.70711891, -0.12678057],
            vec![0.52335543, 0.91745529, 1.66348357],
            vec![-0.52776521, 0.73594303, -1.19771799],
        ];
        let out = dense.forward(&x);
        let expect = vec![
            vec![0.28397459, -0.06502399, 0.34260875],
            vec![0.25160354, -0.07319054, 0.34682375],
            vec![0.3012667, -0.0577441, 0.33831063],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((out[i][j] - expect[i][j]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_dense_forward2() {
        let mut dense = Dense::new(4, 2, 0.0, "0".to_string());
        dense.weights = vec![
            vec![0.00117967, -0.00858846],
            vec![0.01097765, -0.02846899],
            vec![0.02011748, 0.00777664],
            vec![0.00198341, -0.01019557],
        ];
        dense.bias = vec![vec![0.2774197, -0.30174181]];
        let x = vec![
            vec![-0.05650873, 0.18352188, -0.33725816, 0.95372588],
            vec![-1.36199261, -1.27568143, -0.27815827, -0.08191854],
            vec![-0.2615977, 0.50780326, 1.75554298, -0.78205811],
        ];
        let out = dense.forward(&x);
        let expect = vec![
            vec![0.27447452, -0.31882769],
            vec![0.25605069, -0.25505497],
            vec![0.31645154, -0.29232598],
        ];
        for i in 0..3 {
            for j in 0..2 {
                assert!((out[i][j] - expect[i][j]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_dense_backward() {}
}

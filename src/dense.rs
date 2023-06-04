use crate::layer::Layer;
use crate::matrix::{addbias, load_matrix, matadd, matmul, save_matrix, scalar, transpose, Matrix};
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
    pub weights: Matrix,
    pub bias: Matrix,
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
            .map(|v| v.iter().sum())
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
    fn test_dense_backward1() {
        let x = vec![
            vec![-1.40040665, 1.64198078, -0.6429084],
            vec![-1.73607869, 0.64515697, 0.3300737],
            vec![-2.05489541, -0.72318288, 0.41251671],
        ];
        let g = vec![
            vec![-1.35554167, 0.32898185, -0.65938991],
            vec![-2.11199443, 0.63972082, 0.4052747],
            vec![-0.31732176, 0.99892603, 0.38860114],
        ];
        let mut dense = Dense::new(3, 3, 0.001, "0".to_string());
        dense.weights = vec![
            vec![0.00253691, -0.00117961, 0.00237401],
            vec![0.00076495, -0.00629112, -0.01478063],
            vec![0.00271273, 0.01052152, -0.00099406],
        ];
        dense.bias = vec![vec![0.79761756, 1.08705939, -0.84555019]];
        let out = dense.backward(&x, &g);
        let expect_g = vec![
            vec![-0.00539235, 0.00663962, 0.00043964],
            vec![-0.00515043, -0.01163034, 0.0005987],
            vec![-0.00106082, -0.01227087, 0.00926312],
        ];
        let expect_w = vec![
            vec![-0.00368005, 0.0024444, 0.00295272],
            vec![0.00412381, -0.00652162, -0.01367836],
            vec![0.00266926, 0.0101098, -0.00171206],
        ];
        let expect_b = vec![vec![0.80140242, 1.08509177, -0.84568468]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((out[i][j] - expect_g[i][j]).abs() < 1e-4);
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!((dense.weights[i][j] - expect_w[i][j]).abs() < 1e-4);
            }
        }
        for i in 0..3 {
            assert!((dense.bias[0][i] - expect_b[0][i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dense_backward2() {
        let x = vec![
            vec![-0.02154928, -1.3701469],
            vec![0.50650555, -0.46731215],
            vec![0.98467748, 0.15728844],
            vec![-0.00387697, 0.75071803],
        ];
        let g = vec![
            vec![1.26549446, -0.36437136, 0.99830298],
            vec![-0.84536464, 0.18533896, 0.012928],
            vec![0.53164405, 0.17677824, 1.25006833],
            vec![0.22057022, -0.72347588, 1.61007268],
        ];
        let mut dense = Dense::new(3, 3, 0.1, "0".to_string());
        dense.weights = vec![
            vec![0.00194099, -0.01674031, -0.00232448],
            vec![-0.00483631, -0.00068302, -0.00931907],
        ];
        dense.bias = vec![vec![-0.05421712, 0.6576409, 0.16512656]];
        let out = dense.backward(&x, &g);
        let expect_g = vec![
            vec![0.00623546, -0.01517471],
            vec![-0.00477352, 0.00384138],
            vec![-0.00483316, -0.01434141],
            vec![0.00879676, -0.01557698],
        ];
        let expect_w = vec![
            vec![-0.00477806, -0.04460047, -0.12329521],
            vec![0.10412935, 0.00958599, -0.01246594],
        ];
        let expect_b = vec![vec![-0.17145153, 0.7302139, -0.22201064]];
        for i in 0..4 {
            for j in 0..2 {
                assert!((out[i][j] - expect_g[i][j]).abs() < 1e-4);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                assert!((dense.weights[i][j] - expect_w[i][j]).abs() < 1e-4);
            }
        }
        for i in 0..3 {
            assert!((dense.bias[0][i] - expect_b[0][i]).abs() < 1e-4);
        }
    }
}

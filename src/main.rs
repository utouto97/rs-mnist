use rand::prelude::{thread_rng, Distribution};
use rand_distr::Normal;

fn main() {
    println!("Hello, world!");

    let zeros = vec![vec![0.0; 10]; 1];
    let x: Vec<Vec<f32>> = zeros
        .iter()
        .map(|v| v.iter().enumerate().map(|(i, _)| i as f32 - 5.0).collect())
        .collect();

    println!("{} {}", x.len(), x[0].len());
    let mut dense = Dense::new(10, 20, 0.1);
    initialize_matrix(&mut dense.weights);
    let mut relu = Relu::new();
    let x2 = dense.forward(&x);
    for i in &x2 {
        for j in i {
            print!("{} ", j)
        }
        println!()
    }
    let y = relu.forward(&x2);
    for i in &y {
        for j in i {
            print!("{} ", j)
        }
        println!()
    }
    let loss = softmax_cross_entropy(&x2, 0);
    println!("loss: {}", loss);
    let grads0 = grad_softmax_cross_entropy(&x2, 0);
    for i in &grads0 {
        for j in i {
            print!("{} ", j)
        }
        println!()
    }
    let grads = relu.backward(&x2, &x.clone());
    for i in &grads {
        for j in i {
            print!("{} ", j)
        }
        println!()
    }
    let grads2 = dense.backward(&x, &grads);
    for i in grads2 {
        for j in i {
            print!("{} ", j)
        }
        println!()
    }
}

fn softmax_cross_entropy(x: &Matrix, t: usize) -> f32 {
    let sum = x[0].iter().fold(0.0, |acc, e| acc + e.exp());
    -(x[0][t].exp() / sum).log(1.0_f32.exp())
}

fn grad_softmax_cross_entropy(x: &Matrix, t: usize) -> Matrix {
    let sum = x[0].iter().fold(0.0, |acc, e| acc + e.exp());
    let mut softmax = x[0].iter().map(|e| e.exp() / sum).collect::<Vec<_>>();
    softmax[t] -= 1.0;
    vec![softmax; 1]
}

type Matrix = Vec<Vec<f32>>;

fn initialize_matrix(x: &mut Matrix) {
    let mut rng = thread_rng();
    let dist = Normal::<f32>::new(0.0, 1.0).unwrap();
    for i in 0..x.len() {
        for j in 0..x[i].len() {
            x[i][j] = dist.sample(&mut rng);
        }
    }
}

trait Layer {
    fn forward(&self, x: &Matrix) -> Matrix;
    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix;
}

fn scalar(x: &Matrix, k: f32) -> Matrix {
    x.iter()
        .map(|v| v.iter().map(|e| e * k).collect())
        .collect()
}

fn matadd(x: &Matrix, y: &Matrix) -> Matrix {
    x.iter()
        .zip(y.iter())
        .map(|(xs, ys)| xs.iter().zip(ys.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

fn matmul(x: &Matrix, y: &Matrix) -> Matrix {
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

fn transpose(x: &Matrix) -> Matrix {
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

struct Dense {
    lr: f32,
    n_inputs: usize,
    n_outputs: usize,
    weights: Matrix,
    bias: Matrix,
}

impl Dense {
    fn new(n_inputs: usize, n_outputs: usize, lr: f32) -> Self {
        let weights = vec![vec![0.0; n_outputs]; n_inputs];
        let bias = vec![vec![0.0; n_outputs]; 1];
        Self {
            n_inputs,
            n_outputs,
            lr,
            weights,
            bias,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, x: &Matrix) -> Matrix {
        matadd(&matmul(&x, &self.weights), &self.bias)
    }

    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix {
        let grad_input = matmul(&grad_output, &transpose(&self.weights));

        let grad_weights = matmul(&transpose(&x), &grad_output);
        let grad_bias = grad_output
            .iter()
            .map(|v| {
                let sum = v.iter().fold(0.0, |acc, e| acc + e);
                v.iter().map(|_| sum).collect()
            })
            .collect();

        self.weights = matadd(&self.weights, &scalar(&grad_weights, -self.lr));
        self.bias = matadd(&self.bias, &scalar(&grad_bias, -self.lr));

        grad_input
    }
}

struct Relu {}

impl Relu {
    fn new() -> Self {
        Self {}
    }
}

impl Layer for Relu {
    fn forward(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|v| v.iter().map(|e| if *e < 0.0 { 0.0 } else { *e }).collect())
            .collect()
    }

    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix {
        x.iter()
            .zip(grad_output.iter())
            .map(|(xs, gs)| {
                xs.iter()
                    .zip(gs.iter())
                    .map(|(x, g)| if *x < 0.0 { 0.0 } else { *g })
                    .collect()
            })
            .collect()
    }
}

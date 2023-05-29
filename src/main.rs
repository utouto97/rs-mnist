use rand::prelude::{thread_rng, Distribution};
use rand_distr::Normal;

fn main() {
    println!("Hello, world!");

    let mut rng = thread_rng();
    let dist = Normal::<f32>::new(0.0, 1.0).unwrap();
    println!("{}", dist.sample(&mut rng));

    let zeros = vec![vec![0.0; 10]; 10];
    let x: Vec<Vec<f32>> = zeros
        .iter()
        .map(|v| v.iter().enumerate().map(|(i, _)| i as f32 - 5.0).collect())
        .collect();
    let relu = Relu::new();

    let grads = relu.backward(&x, x.clone());
    for i in grads {
        for j in i {
            print!("{} ", j)
        }
        println!()
    }
}

type Matrix = Vec<Vec<f32>>;

trait Layer {
    fn forward(&self, x: &Matrix) -> Matrix;
    fn backward(&self, x: &Matrix, grad_output: Matrix) -> Matrix;
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

    fn backward(&self, x: &Matrix, grad_output: Matrix) -> Matrix {
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

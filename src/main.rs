use std::fs::File;
use std::io::Read;

use rand::{
    prelude::{thread_rng, Distribution},
    seq::SliceRandom,
    Rng,
};
use rand_distr::Normal;

fn main() {
    const LR: f32 = 0.1;
    let (mut x_train, mut y_train) = load_train_datasets();
    shuffle(&mut x_train, &mut y_train);
    for (x, y) in BatchIter::new(&x_train, &y_train, 250) {
        println!("x {} {} ", x.len(), x[0].len());
        println!("y {}", y.len());

        let mut network = Network::new();
        network.add_layer(Box::new(Dense::new(28 * 28, 10, LR)));
        // network.add_layer(Box::new(Sigmoid::new()));
        // network.add_layer(Box::new(Dense::new(100, 200, LR)));
        // network.add_layer(Box::new(Sigmoid::new()));
        // network.add_layer(Box::new(Dense::new(200, 10, LR)));

        let y = network.forward(&x);
        // for (idx, o) in network.outputs.into_iter().enumerate() {
        //     for i in &o {
        //         for j in i {
        //             if idx == 0 {
        //                 print!("{:>02.0} ", j * 10.0);
        //             } else {
        //                 print!("{} ", j);
        //             }
        //         }
        //         println!();
        //     }
        //     println!();
        // }
        let loss = softmax_cross_entropy(&y, y_train[0]);
        println!("loss {}", loss);
        println!("predict {}", argmax(&y));
        break;
    }
}

fn shuffle(images: &mut Matrix, labels: &mut Vec<usize>) {
    let mut rng = rand::thread_rng();
    for i in 0..labels.len() {
        let j = rng.gen_range(0..labels.len());
        let tmp = images[i].clone();
        images[i] = images[j].clone();
        images[j] = tmp.to_vec();
    }
}

struct BatchIter<'a> {
    images: &'a Matrix,
    labels: &'a Vec<usize>,
    batch_size: usize,
    curr: usize,
}

impl<'a> BatchIter<'a> {
    fn new(images: &'a Matrix, labels: &'a Vec<usize>, batch_size: usize) -> Self {
        let curr = 0;
        Self {
            images,
            labels,
            batch_size,
            curr,
        }
    }
}

impl<'a> Iterator for BatchIter<'a> {
    type Item = (Matrix, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.curr;
        let end = start + self.batch_size;
        if end > self.labels.len() {
            return None;
        }
        self.curr = end;
        Some((
            self.images[start..end].to_vec(),
            self.labels[start..end].to_vec(),
        ))
    }
}

struct Network {
    layers: Vec<Box<dyn Layer>>,
    outputs: Vec<Matrix>,
}

impl Network {
    fn new() -> Self {
        Self {
            layers: vec![],
            outputs: vec![],
        }
    }

    fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer)
    }

    fn forward(&mut self, x: &Matrix) -> Matrix {
        println!("---");
        self.outputs.push(x.to_vec());
        for layer in &self.layers {
            let nx = &layer.forward(&self.outputs.last().unwrap());
            self.outputs.push(nx.to_vec());
        }
        self.outputs.last().unwrap().to_vec()
    }
}

fn loadfile(fname: &str) -> Vec<u8> {
    let mut f = File::open(fname).unwrap();
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf).unwrap();
    buf
}

fn load_train_datasets() -> (Matrix, Vec<usize>) {
    let images_u8 = loadfile("datasets/train-images-idx3-ubyte")[16..].to_vec();
    let labels_u8 = loadfile("datasets/train-labels-idx1-ubyte")[8..].to_vec();

    let normalized_images = images_u8
        .into_iter()
        .map(|e| ((e as f32) - 127.0) / 255.0)
        .collect::<Vec<_>>();
    let n = normalized_images.len() / (28 * 28);
    let mut images: Matrix = vec![];
    for k in 0..n {
        let base = k * 28 * 28;
        images.push(normalized_images[base..(base + 28 * 28)].to_vec());
    }
    let labels = labels_u8
        .into_iter()
        .map(|e| (e as usize))
        .collect::<Vec<_>>();
    (images, labels)
}

fn argmax(x: &Matrix) -> usize {
    let mut res = 0;
    let mut max = x[0][0];
    for i in 1..x[0].len() {
        if max < x[0][i] {
            res = i;
            max = x[0][i];
        }
    }
    res
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
        let mut weights = vec![vec![0.0; n_outputs]; n_inputs];
        initialize_matrix(&mut weights);
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

struct Sigmoid {}

impl Sigmoid {
    fn new() -> Self {
        Self {}
    }
}

impl Layer for Sigmoid {
    fn forward(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|v| v.into_iter().map(|e| 1.0 / (1.0 + (-e).exp())).collect())
            .collect()
    }

    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix {
        x.iter()
            .zip(grad_output.iter())
            .map(|(xs, gs)| {
                xs.into_iter()
                    .zip(gs.into_iter())
                    .map(|(x, g)| {
                        let fx = 1.0 / (1.0 + (-x).exp());
                        g * fx * (1.0 - fx)
                    })
                    .collect()
            })
            .collect()
    }
}

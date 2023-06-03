use std::fs::File;
use std::io::Read;
use std::io::{stdout, Write};

use rand::{
    prelude::{thread_rng, Distribution},
    Rng,
};
use rand_distr::Normal;

fn main() {
    const LR: f32 = 0.001;
    const BATCH_SIZE: usize = 5000;
    const EPOCHS: usize = 10;

    let mut network = Network::new();
    network.add_layer(Box::new(Dense::new(28 * 28, 100, LR)));
    network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(Dense::new(100, 200, LR)));
    network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(Dense::new(200, 10, LR)));

    let (mut x_train, mut y_train) = load_train_datasets();
    shuffle(&mut x_train, &mut y_train);
    for epoch in 0..EPOCHS {
        println!("{} / {}", epoch, EPOCHS);
        let iters = y_train.len() / BATCH_SIZE;
        let mut acc = 0.0;
        let mut total_loss = 0.0;
        for (_i, (x, y)) in BatchIter::new(&x_train, &y_train, BATCH_SIZE).enumerate() {
            let out = network.forward(&x);
            let loss = softmax_cross_entropy(&out, &y);
            let gy = grad_softmax_cross_entropy(&out, &y);
            network.backward(&gy);

            total_loss += loss;
            let preds = argmax(&out);
            acc += accuracy(&preds, &y);
            print!("*");
            stdout().flush().unwrap();
        }
        println!();
        println!("loss {:4.4}", total_loss / iters as f32);
        println!("acc  {:4.4}", acc / iters as f32);
    }
}

fn argmax(x: &Matrix) -> Vec<usize> {
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

fn accuracy(pred: &Vec<usize>, t: &Vec<usize>) -> f32 {
    let n = pred.len() as f32;
    pred.into_iter().zip(t.into_iter()).fold(
        0.0,
        |acc, (p, t)| {
            if p == t {
                acc + 1.0
            } else {
                acc
            }
        },
    ) / n
}

fn shuffle(images: &mut Matrix, labels: &mut Vec<usize>) {
    let mut rng = rand::thread_rng();
    for i in 0..labels.len() {
        let j = rng.gen_range(0..labels.len());
        images.swap(i, j);
        labels.swap(i, j);
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
    inputs: Vec<Matrix>,
}

impl Network {
    fn new() -> Self {
        Self {
            layers: vec![],
            inputs: vec![],
        }
    }

    fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer)
    }

    fn forward(&mut self, x: &Matrix) -> Matrix {
        self.inputs.clear();
        self.inputs.push(x.clone());
        for layer in &self.layers {
            self.inputs
                .push(layer.forward(&self.inputs.last().unwrap()));
        }
        self.inputs.last().unwrap().to_vec()
    }

    fn backward(&mut self, g: &Matrix) {
        let n = self.layers.len();
        let mut gs = vec![g.clone()];
        for i in (0..n).rev() {
            gs.push(self.layers[i].backward(&self.inputs[i], gs.last().unwrap()));
        }
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
        .map(|e| (e as f32) / 255.0)
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

fn softmax_cross_entropy(x: &Matrix, t: &Vec<usize>) -> f32 {
    let n = x.len() as f32;
    x.iter()
        .zip(t.into_iter())
        .map(|(x, t)| {
            let sum = x.iter().map(|e| e.exp()).sum::<f32>();
            -(x[*t].exp() / sum).ln()
        })
        .sum::<f32>()
        / n
}

fn grad_softmax_cross_entropy(x: &Matrix, t: &Vec<usize>) -> Matrix {
    x.iter()
        .zip(t.into_iter())
        .map(|(x, t)| {
            let sum = x.iter().map(|e| e.exp()).sum::<f32>();
            let mut softmax = x.iter().map(|e| e.exp() / sum).collect::<Vec<_>>();
            softmax[*t] -= 1.0;
            softmax
        })
        .collect()
}

type Matrix = Vec<Vec<f32>>;

fn initialize_matrix(x: &mut Matrix) {
    let mut rng = thread_rng();
    let dist = Normal::<f32>::new(0.0, 0.01).unwrap();
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

fn addbias(x: &Matrix, y: &Matrix) -> Matrix {
    x.iter()
        .map(|xs| xs.iter().zip(y[0].iter()).map(|(x, y)| x + y).collect())
        .collect()
}

fn matadd(x: &Matrix, y: &Matrix) -> Matrix {
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
    weights: Matrix,
    bias: Matrix,
}

impl Dense {
    fn new(n_inputs: usize, n_outputs: usize, lr: f32) -> Self {
        let mut weights = vec![vec![0.0; n_outputs]; n_inputs];
        initialize_matrix(&mut weights);
        let bias = vec![vec![0.0; n_outputs]; 1];
        Self { lr, weights, bias }
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
}

struct Sigmoid {}

impl Sigmoid {
    fn new() -> Self {
        Self {}
    }

    fn _sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Layer for Sigmoid {
    fn forward(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|v| v.into_iter().map(|e| self._sigmoid(*e)).collect())
            .collect()
    }

    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix {
        x.iter()
            .zip(grad_output.iter())
            .map(|(xs, gs)| {
                xs.into_iter()
                    .zip(gs.into_iter())
                    .map(|(x, g)| {
                        let fx = self._sigmoid(*x);
                        g * fx * (1.0 - fx)
                    })
                    .collect()
            })
            .collect()
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
}

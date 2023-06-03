mod dense;
mod layer;
mod matrix;
mod network;
mod relu;

use crate::dense::Dense;
use crate::layer::Layer;
use crate::matrix::{
    addbias, argmax, load_matrix, matadd, matmul, save_matrix, scalar, transpose, Matrix,
};
use crate::network::Network;
use crate::relu::Relu;

use std::fs::File;
use std::io::Read;
use std::io::{stdout, Write};

use rand::Rng;

fn main() {
    const LR: f32 = 0.01;
    const BATCH_SIZE: usize = 5000;
    const EPOCHS: usize = 10;

    let mut network = Network::new();
    network.add_layer(Box::new(Dense::new(28 * 28, 100, LR, "0".to_string())));
    network.add_layer(Box::new(Relu::new()));
    // network.add_layer(Box::new(Dense::new(100, 200, LR, "1".to_string())));
    // network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(Dense::new(100, 10, LR, "2".to_string())));

    let (mut x_train, mut y_train) = load_datasets(true);
    shuffle(&mut x_train, &mut y_train);
    let (x_test, y_test) = load_datasets(false);
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

        {
            let out = network.forward(&x_test);
            let preds = argmax(&out);
            let acc = accuracy(&preds, &y_test);
            println!("acc  {:4.4}", acc);
        }
    }

    network.save();
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

fn loadfile(fname: &str) -> Vec<u8> {
    let mut f = File::open(fname).unwrap();
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf).unwrap();
    buf
}

fn load_datasets(train: bool) -> (Matrix, Vec<usize>) {
    let (images_u8, labels_u8) = if train {
        (
            loadfile("datasets/train-images-idx3-ubyte")[16..].to_vec(),
            loadfile("datasets/train-labels-idx1-ubyte")[8..].to_vec(),
        )
    } else {
        (
            loadfile("datasets/t10k-images-idx3-ubyte")[16..].to_vec(),
            loadfile("datasets/t10k-labels-idx1-ubyte")[8..].to_vec(),
        )
    };

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
    let n = t.len() as f32;
    x.iter()
        .zip(t.into_iter())
        .map(|(x, t)| {
            let sum = x.iter().map(|e| e.exp()).sum::<f32>();
            let mut softmax = x.iter().map(|e| e.exp() / sum / n).collect::<Vec<_>>();
            softmax[*t] -= 1.0 / n;
            softmax
        })
        .collect()
}

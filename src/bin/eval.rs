extern crate rs_mnist;

use rs_mnist::dense::Dense;
use rs_mnist::matrix::{argmax, Matrix};
use rs_mnist::network::Network;
use rs_mnist::relu::Relu;

use std::fs::File;
use std::io::Read;

fn main() {
    const LR: f32 = 0.001;

    let mut network = Network::new();
    network.add_layer(Box::new(Dense::new(28 * 28, 100, LR, "0".to_string())));
    network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(Dense::new(100, 200, LR, "1".to_string())));
    network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(Dense::new(200, 10, LR, "2".to_string())));
    network.load();

    let (x_test, y_test) = load_datasets(false);

    let out = network.forward(&x_test);
    let preds = argmax(&out);
    let acc = accuracy(&preds, &y_test);
    println!("test  acc  {:4.4}", acc);
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
        .map(|e| (e as f32))
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

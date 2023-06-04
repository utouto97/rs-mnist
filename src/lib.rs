pub mod dense;
pub mod layer;
pub mod loss;
pub mod matrix;
pub mod network;
pub mod relu;

use crate::dense::Dense;
use crate::matrix::argmax;
use crate::network::Network;
use crate::relu::Relu;
use matrix::Matrix;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn predict(
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
    inputs: Vec<f32>,
) -> usize {
    const LR: f32 = 0.001;
    let mut dense1 = Dense::new(28 * 28, 100, LR, "0".to_string());
    copy(&mut dense1.weights, &w1);
    copy(&mut dense1.bias, &b1);
    let mut dense2 = Dense::new(100, 200, LR, "1".to_string());
    copy(&mut dense2.weights, &w2);
    copy(&mut dense2.bias, &b2);
    let mut dense3 = Dense::new(200, 10, LR, "2".to_string());
    copy(&mut dense3.weights, &w3);
    copy(&mut dense3.bias, &b3);

    let mut network = Network::new();
    network.add_layer(Box::new(dense1));
    network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(dense2));
    network.add_layer(Box::new(Relu::new()));
    network.add_layer(Box::new(dense3));

    let x = vec![inputs];
    // alert(format!("{:?}", x).as_str());
    let out = network.forward(&x);
    // alert(format!("{:?}", out).as_str());
    let preds = argmax(&out);
    // alert(format!("{:?}", preds).as_str());
    preds[0]
}

fn copy(x: &mut Matrix, y: &Vec<f32>) {
    let r = x.len();
    let c = x[0].len();
    for i in 0..r {
        for j in 0..c {
            x[i][j] = y[i * c + j];
        }
    }
}

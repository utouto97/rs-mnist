use crate::{Layer, Matrix};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    inputs: Vec<Matrix>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            layers: vec![],
            inputs: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer)
    }

    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        self.inputs.clear();
        self.inputs.push(x.clone());
        for layer in &self.layers {
            self.inputs
                .push(layer.forward(&self.inputs.last().unwrap()));
        }
        self.inputs.last().unwrap().to_vec()
    }

    pub fn backward(&mut self, g: &Matrix) {
        let n = self.layers.len();
        let mut gs = vec![g.clone()];
        for i in (0..n).rev() {
            gs.push(self.layers[i].backward(&self.inputs[i], gs.last().unwrap()));
        }
    }

    pub fn save(&self) {
        for layer in &self.layers {
            layer.save()
        }
    }

    pub fn load(&mut self) {
        for i in 0..self.layers.len() {
            self.layers[i].load()
        }
    }
}

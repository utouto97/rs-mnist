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

#[cfg(test)]
mod tests {
    use crate::{dense::Dense, loss::grad_softmax_cross_entropy, relu::Relu};

    use super::*;

    #[test]
    fn test_network() {
        let x = vec![
            vec![
                -0.50137948,
                0.90610786,
                0.0629102,
                0.12018931,
                1.32041643,
                -0.01351908,
                0.13926187,
                1.63817711,
            ],
            vec![
                0.97857469,
                -1.13066685,
                0.14594583,
                0.32020487,
                -0.7474786,
                -0.84103068,
                -1.44756796,
                -0.80640032,
            ],
            vec![
                2.49657838,
                1.66390973,
                -0.89682964,
                -2.11580415,
                0.44182283,
                1.32320713,
                -1.04325849,
                -1.18465581,
            ],
        ];
        let outs = vec![
            vec![
                vec![
                    -0.01129225,
                    -0.03821541,
                    0.0169413,
                    0.00366862,
                    -0.02250481,
                    -0.00295436,
                    0.02696538,
                    -0.01073435,
                    -0.02432323,
                    -0.0185001,
                ],
                vec![
                    0.00975089,
                    0.02512429,
                    -0.03493351,
                    -0.01875033,
                    0.02280609,
                    -0.01049339,
                    -0.01624625,
                    0.02463949,
                    0.04411266,
                    -0.00530322,
                ],
                vec![
                    -0.07567336,
                    0.09928554,
                    0.01797975,
                    0.03606872,
                    -0.0494089,
                    0.05462047,
                    -0.0809059,
                    -0.04216917,
                    -0.0297297,
                    -0.020728,
                ],
            ],
            vec![
                vec![
                    0., 0., 0.0169413, 0.00366862, 0., 0., 0.02696538, 0., 0., 0.,
                ],
                vec![
                    0.00975089, 0.02512429, 0., 0., 0.02280609, 0., 0., 0.02463949, 0.04411266, 0.,
                ],
                vec![
                    0., 0.09928554, 0.01797975, 0.03606872, 0., 0.05462047, 0., 0., 0., 0.,
                ],
            ],
            vec![
                vec![-0.00024223, 0.00053081, 0.00039536, 0.00045668, 0.000253],
                vec![0.00038307, -0.00169088, 0.00065873, 0.00043046, 0.00162302],
                vec![
                    -0.00028888,
                    -0.0005926,
                    -0.00062918,
                    0.00194026,
                    -0.00042538,
                ],
            ],
        ];
        let mut dense1 = Dense::new(8, 10, 0.001, "0".to_string());
        dense1.weights = vec![
            vec![
                -1.35157273e-02,
                1.22371241e-02,
                -2.00097497e-03,
                -6.27534299e-03,
                -3.15169182e-03,
                4.63738104e-03,
                -1.53294550e-02,
                3.07840066e-03,
                3.28205251e-03,
                1.48625525e-03,
            ],
            vec![
                -9.55776493e-03,
                3.83999854e-04,
                1.59538773e-02,
                1.47225294e-02,
                -1.00444636e-02,
                1.46562133e-02,
                -4.33416468e-03,
                1.95917127e-03,
                -1.82758207e-02,
                2.83734312e-04,
            ],
            vec![
                4.34187102e-03,
                -9.87049365e-03,
                -9.47788544e-03,
                -6.61493465e-03,
                6.97041961e-03,
                -1.06284845e-02,
                9.16373806e-03,
                2.35173869e-02,
                4.51165138e-03,
                5.78509333e-03,
            ],
            vec![
                9.91126534e-03,
                -2.72941865e-03,
                4.03845270e-03,
                -2.73411077e-03,
                -5.02340784e-04,
                2.54185762e-03,
                1.30804691e-02,
                1.02271318e-02,
                8.77277376e-04,
                9.61008873e-03,
            ],
            vec![
                -1.22286278e-03,
                9.10540985e-03,
                5.30256271e-03,
                2.20124185e-03,
                -1.47204405e-02,
                4.35728269e-03,
                1.57828444e-02,
                -1.68430716e-02,
                -1.85458574e-03,
                -3.17886927e-03,
            ],
            vec![
                -4.91013886e-03,
                9.03690139e-03,
                -2.75579178e-04,
                1.90936816e-03,
                -7.07231402e-03,
                7.03449358e-04,
                -3.43606187e-03,
                -1.44639771e-03,
                -7.71561823e-03,
                2.53039837e-03,
            ],
            vec![
                9.21359536e-04,
                -6.49058755e-03,
                9.99206913e-03,
                -1.28194817e-03,
                9.53828590e-04,
                2.44371411e-03,
                1.94307383e-03,
                -5.46436746e-03,
                -6.79711456e-03,
                1.31524956e-02,
            ],
            vec![
                -5.77029941e-03,
                -2.59286849e-02,
                -4.15329413e-03,
                -9.01939799e-03,
                2.34827815e-03,
                -1.19831224e-02,
                -6.03541790e-05,
                5.68100828e-03,
                -1.96314547e-03,
                -1.04573461e-02,
            ],
        ];
        let mut dense2 = Dense::new(10, 5, 0.001, "0".to_string());
        dense2.weights = vec![
            vec![0.00281244, -0.00803539, -0.01127738, 0.01069142, 0.00357014],
            vec![
                0.00138047,
                -0.00774434,
                -0.00658416,
                0.01319422,
                -0.00199744,
            ],
            vec![-0.00775504, -0.00772245, 0.00126468, 0.01692756, 0.00444527],
            vec![0.00300643, -0.00909757, -0.00978695, 0.00340596, 0.00345795],
            vec![0.01936285, -0.01450902, -0.00155485, 0.00124209, 0.00229708],
            vec![-0.00723066, 0.01177732, 0.00649568, 0.00371771, -0.0079039],
            vec![-0.00451974, 0.02577434, 0.01519865, 0.00583756, 0.00611927],
            vec![-0.00166168, -0.00686825, 0.01097726, 0.01897607, 0.01301431],
            vec![
                -0.00180651,
                -0.02080649,
                0.01584801,
                -0.01136121,
                0.02868421,
            ],
            vec![-0.00925911, 0.0009818, -0.00750133, 0.0194073, -0.01257454],
        ];

        let mut network = Network::new();
        network.add_layer(Box::new(dense1));
        network.add_layer(Box::new(Relu::new()));
        network.add_layer(Box::new(dense2));
        let out = network.forward(&x);
        assert_eq!(network.inputs.len(), 4);
        for k in 0..3 {
            for i in 0..outs[k].len() {
                for j in 0..outs[k][i].len() {
                    assert!((outs[k][i][j] - network.inputs[k + 1][i][j]).abs() < 1e-4);
                }
            }
        }

        let g = grad_softmax_cross_entropy(&out, &vec![1, 2, 3]);
        network.backward(&g);
        let out2 = network.forward(&x);
        assert_eq!(network.inputs.len(), 4);
        assert!((out2[0][0] - -4.42309346e-04).abs() < 1e-4);
    }
}

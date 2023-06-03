use crate::Matrix;

pub fn softmax_cross_entropy(x: &Matrix, t: &Vec<usize>) -> f32 {
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

pub fn grad_softmax_cross_entropy(x: &Matrix, t: &Vec<usize>) -> Matrix {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_cross_entropy1() {
        let x = vec![
            vec![0.45302878, -0.4208616, -0.71232351],
            vec![0.20674101, -1.39604046, -0.36633254],
            vec![-0.20416752, 0.37670643, 1.29414668],
        ];
        let t = vec![0, 1, 2];
        let expect = 1.0676438210940964;
        assert!((softmax_cross_entropy(&x, &t) - expect).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_cross_entropy2() {
        let x = vec![
            vec![
                0.02127046,
                0.18419325,
                -0.11955557,
                1.19153021,
                1.54618452,
                -0.41638406,
                1.3391051,
                0.13134751,
                -0.31064525,
                -0.1992153,
            ],
            vec![
                -0.06212451,
                -0.55091702,
                0.53401083,
                0.21627452,
                0.60599668,
                1.27962263,
                -0.80425168,
                -0.38209024,
                -0.17642485,
                0.98455581,
            ],
            vec![
                -0.95777901,
                -0.90862446,
                -1.59025084,
                -0.81824309,
                0.04832966,
                1.51607963,
                -1.42664232,
                0.95780805,
                0.94387061,
                -1.44341619,
            ],
            vec![
                0.18225517,
                0.10494158,
                -0.42536406,
                -1.80011312,
                -0.78888738,
                -0.22218121,
                -0.21873344,
                0.40491421,
                -0.1868413,
                -0.72378924,
            ],
            vec![
                1.21478715,
                -0.13589561,
                -1.06897553,
                1.5392096,
                0.804061,
                0.12979832,
                -0.71269309,
                -0.06919108,
                -0.68735221,
                -1.23856806,
            ],
        ];
        let t = vec![0, 1, 2, 4, 5];
        let expect = 3.13559460748951;
        assert!((softmax_cross_entropy(&x, &t) - expect).abs() < 1e-4);
    }

    #[test]
    fn test_grad_softmax_cross_entropy() {
        let x = vec![
            vec![
                0.02127046,
                0.18419325,
                -0.11955557,
                1.19153021,
                1.54618452,
                -0.41638406,
                1.3391051,
                0.13134751,
                -0.31064525,
                -0.1992153,
            ],
            vec![
                -0.06212451,
                -0.55091702,
                0.53401083,
                0.21627452,
                0.60599668,
                1.27962263,
                -0.80425168,
                -0.38209024,
                -0.17642485,
                0.98455581,
            ],
            vec![
                -0.95777901,
                -0.90862446,
                -1.59025084,
                -0.81824309,
                0.04832966,
                1.51607963,
                -1.42664232,
                0.95780805,
                0.94387061,
                -1.44341619,
            ],
            vec![
                0.18225517,
                0.10494158,
                -0.42536406,
                -1.80011312,
                -0.78888738,
                -0.22218121,
                -0.21873344,
                0.40491421,
                -0.1868413,
                -0.72378924,
            ],
            vec![
                1.21478715,
                -0.13589561,
                -1.06897553,
                1.5392096,
                0.804061,
                0.12979832,
                -0.71269309,
                -0.06919108,
                -0.68735221,
                -1.23856806,
            ],
        ];
        let t = vec![0, 1, 2, 4, 5];
        let expect = vec![
            vec![
                -0.18881437,
                0.01316488,
                0.00971629,
                0.03604939,
                0.05139517,
                0.00722087,
                0.04178197,
                0.01248724,
                0.00802623,
                0.00897232,
            ],
            vec![
                0.01292975,
                -0.19206932,
                0.02346866,
                0.01708037,
                0.02522037,
                0.04946562,
                0.00615585,
                0.00938925,
                0.01153321,
                0.03682625,
            ],
            vec![
                0.00604899,
                0.00635376,
                -0.19678631,
                0.00695477,
                0.01654361,
                0.07179035,
                0.00378493,
                0.04107824,
                0.04050968,
                0.00372197,
            ],
            vec![
                0.02999165,
                0.02776026,
                0.01633483,
                0.00413113,
                -0.18864366,
                0.02001502,
                0.02008414,
                0.03747141,
                0.02073499,
                0.01212022,
            ],
            vec![
                0.04542388,
                0.01176766,
                0.0046287,
                0.06283174,
                0.03012369,
                -0.18465102,
                0.00660983,
                0.01257939,
                0.00677947,
                0.00390666,
            ],
        ];
        let out = grad_softmax_cross_entropy(&x, &t);
        for i in 0..5 {
            for j in 0..10 {
                assert!((out[i][j] - expect[i][j]).abs() < 1e-4);
            }
        }
    }
}

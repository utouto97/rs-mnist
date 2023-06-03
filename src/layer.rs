use crate::Matrix;

pub trait Layer {
    fn forward(&self, x: &Matrix) -> Matrix;
    fn backward(&mut self, x: &Matrix, grad_output: &Matrix) -> Matrix;
    fn save(&self);
    fn load(&mut self);
}

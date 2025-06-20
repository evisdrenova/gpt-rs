pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error>;
}

pub struct ModuleList<T: Module> {
    layers: Vec<T>,
}

impl<T: Module> ModuleList<T> {
    pub fn new() -> Self {
        ModuleList { layers: Vec::new() }
    }
}

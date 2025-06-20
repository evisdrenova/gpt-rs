use candle_core::{Error, Tensor};

use crate::{attention::CausalAttention, layers::Linear};

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

    pub fn with_capacity(capacity: usize) -> Self {
        ModuleList {
            layers: Vec::with_capacity(capacity),
        }
    }

    /// Create a ModuleList from a vector of modules
    pub fn from_vec(layers: Vec<T>) -> Self {
        ModuleList { layers }
    }

    /// Add a module to the list
    pub fn push(&mut self, module: T) {
        self.layers.push(module);
    }

    /// Remove and return the last module
    pub fn pop(&mut self) -> Option<T> {
        self.layers.pop()
    }

    /// Insert a module at a specific index
    pub fn insert(&mut self, index: usize, module: T) {
        self.layers.insert(index, module);
    }

    /// Remove a module at a specific index
    pub fn remove(&mut self, index: usize) -> T {
        self.layers.remove(index)
    }

    /// Get a reference to a module by index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.layers.get(index)
    }

    /// Iterate over all modules
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.layers.iter()
    }

    /// Iterate over all modules mutably
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.layers.iter_mut()
    }

    /// Apply all modules sequentially
    pub fn forward_sequential(&self, mut input: Tensor) -> Result<Tensor, Error> {
        for module in &self.layers {
            input = module.forward(&input)?;
        }
        Ok(input)
    }

    /// Apply all modules in parallel and return all outputs
    pub fn forward_parallel(&self, input: &Tensor) -> Result<Vec<Tensor>, Error> {
        let mut outputs = Vec::with_capacity(self.layers.len());
        for module in &self.layers {
            outputs.push(module.forward(input)?);
        }
        Ok(outputs)
    }
}

// Implement IntoIterator for owned iteration
impl<T: Module> IntoIterator for ModuleList<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}

// Implement IntoIterator for borrowed iteration
impl<'a, T: Module> IntoIterator for &'a ModuleList<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter()
    }
}

// Implement IntoIterator for mutable borrowed iteration
impl<'a, T: Module> IntoIterator for &'a mut ModuleList<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}

impl Module for CausalAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Your existing forward implementation
        self.forward(input)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Your existing Linear forward implementation
        self.forward(input)
    }
}

mod lowering;
mod emitting;

use std::collections::HashMap;
use lowering::lcr::Crate;
pub use lowering::CrateSource;
pub use lowering::lcr;
#[macro_use]
pub use lowering::path;


#[derive(Debug, Clone, Default)]
pub struct Compiler {
    crates: HashMap<(Box<str>, (u16, u16, u16)), Crate>,
}

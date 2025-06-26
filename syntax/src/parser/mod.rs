pub mod ast;

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(lang, "/src/parser/lang.rs");

pub use lang::{ModuleParser, ExpressionParser};

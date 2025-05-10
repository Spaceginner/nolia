use lalrpop_util::lalrpop_mod;

lalrpop_mod!(lang, "/parser/lang.rs");

pub mod ast;

pub use lang::{ModuleParser, InlineStatementBlockParser, ExpressionParser};

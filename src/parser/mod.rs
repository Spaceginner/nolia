use lalrpop_util::lalrpop_mod;

lalrpop_mod!(lang, "/parser/lang.rs");

pub mod ast;
mod lexer;

pub use lexer::Lexer;
pub use lang::{ModuleParser, ExpressionParser};

pub type Error<'s> = lalrpop_util::ParseError<lexer::Loc, lexer::Token<'s>, lexer::Error>;

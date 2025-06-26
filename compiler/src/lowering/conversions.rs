use std::num::NonZero;
use syntax::parser::ast;
use super::lcr;

impl<'p, 's> From<&'p [&'s str]> for lcr::Path {
    fn from(path: &'p [&'s str]) -> Self {
        Self { parts: path.iter().map(|&p| p.into()).collect(), crate_: None }
    }
}

impl From<ast::IntegerTypeSize> for lcr::IntegerTypeSize {
    fn from(value: ast::IntegerTypeSize) -> Self {
        match value {
            ast::IntegerTypeSize::Byte => Self::Byte,
            ast::IntegerTypeSize::Word => Self::Word,
            ast::IntegerTypeSize::DWord => Self::DWord,
            ast::IntegerTypeSize::QWord => Self::QWord,
            ast::IntegerTypeSize::OWord => Self::OWord,
        }
    }
}


impl From<ast::PrimitiveType> for lcr::PrimitiveType {
    fn from(value: ast::PrimitiveType) -> Self {
        match value {
            ast::PrimitiveType::String => Self::String,
            ast::PrimitiveType::Char => Self::Char,
            ast::PrimitiveType::Bool => Self::Bool,
            ast::PrimitiveType::Integer { signed, size } => Self::Integer { signed, size: size.into() },
            ast::PrimitiveType::Void => Self::Void,
        }
    }
}


impl From<lcr::Path> for lcr::ConcreteTypeId {
    fn from(path: lcr::Path) -> Self {
        Self { path }
    }
}


impl From<lcr::ConcreteTypeId> for lcr::ConcreteTypeRef {
    fn from(id: lcr::ConcreteTypeId) -> Self {
        Self { id, generics: vec![] }
    }
}


impl From<ast::Integer> for lcr::LiteralInteger {
    fn from(value: ast::Integer) -> Self {
        match value {
            ast::Integer::I8(n) => Self::I8(n),
            ast::Integer::I16(n) => Self::I16(n),
            ast::Integer::I32(n) => Self::I32(n),
            ast::Integer::I64(n) => Self::I64(n),
            ast::Integer::I128(n) => Self::I128(n),
            ast::Integer::U8(n) => Self::U8(n),
            ast::Integer::U16(n) => Self::U16(n),
            ast::Integer::U32(n) => Self::U32(n),
            ast::Integer::U64(n) => Self::U64(n),
            ast::Integer::U128(n) => Self::U128(n),
        }
    }
}


impl From<lcr::LiteralInteger> for vm::Integer {
    fn from(int: lcr::LiteralInteger) -> Self {
        match int {
            lcr::LiteralInteger::I8(n) => vm::Integer::I8(n),
            lcr::LiteralInteger::I16(n) => vm::Integer::I16(n),
            lcr::LiteralInteger::I32(n) => vm::Integer::I32(n),
            lcr::LiteralInteger::I64(n) => vm::Integer::I64(n),
            lcr::LiteralInteger::I128(n) => vm::Integer::I128(n),
            lcr::LiteralInteger::U8(n) => vm::Integer::U8(n),
            lcr::LiteralInteger::U16(n) => vm::Integer::U16(n),
            lcr::LiteralInteger::U32(n) => vm::Integer::U32(n),
            lcr::LiteralInteger::U64(n) => vm::Integer::U64(n),
            lcr::LiteralInteger::U128(n) => vm::Integer::U128(n),
        }
    }
}

impl From<ast::LiteralExpression<'_>> for lcr::LiteralValue {
    fn from(value: ast::LiteralExpression) -> Self {
        match value {
            ast::LiteralExpression::Integer(n) => Self::Integer(n.into()),
            ast::LiteralExpression::Float(f) => Self::Float(f),
            ast::LiteralExpression::Char(c) => Self::Char(c),
            ast::LiteralExpression::String(s) => Self::String(s.into()),
            ast::LiteralExpression::Bool(b) => Self::Bool(b),
            ast::LiteralExpression::Void => Self::Void,
        }
    }
}

impl From<lcr::LiteralValue> for vm::ConstItem {
    fn from(value: lcr::LiteralValue) -> Self {
        match value {
            lcr::LiteralValue::Integer(n) => vm::ConstItem::Integer(n.into()),
            lcr::LiteralValue::Float(f) => vm::ConstItem::Float(f),
            lcr::LiteralValue::Char(c) => vm::ConstItem::Char(c),
            lcr::LiteralValue::String(s) => vm::ConstItem::String(s.to_string()),
            _ => unreachable!("not supported ({value:?})"),
        }
    }
}


impl From<ast::AsmId> for lcr::AsmId {
    fn from(ast::AsmId { space, item }: ast::AsmId) -> Self {
        Self { item, space: NonZero::new(space) }
    }
}


impl From<lcr::AsmId> for vm::Id {
    fn from(lcr::AsmId { space, item }: lcr::AsmId) -> Self {
        Self { space, item }
    }
}


impl From<lcr::AsmId> for vm::SysCallId {
    fn from(asm_id: lcr::AsmId) -> Self {
        Self::try_from(vm::Id::from(asm_id)).unwrap()
    }
}


impl From<lcr::AsmId> for vm::SysItemId {
    fn from(asm_id: lcr::AsmId) -> Self {
        Self::try_from(vm::Id::from(asm_id)).unwrap()
    }
}

impl From<lcr::Instruction> for lcr::SBlock {
    fn from(instr: lcr::Instruction) -> Self {
        Self {
            label: None,
            tag: Box::new(lcr::SBlockTag::Simple {
                closed: false,
                decls: vec![],
                code: vec![instr],
            }),
        }
    }
}


impl From<Vec<lcr::Instruction>> for lcr::SBlock {
    fn from(instrs: Vec<lcr::Instruction>) -> Self {
        Self {
            label: None,
            tag: Box::new(lcr::SBlockTag::Simple {
                closed: false,
                decls: vec![],
                code: instrs,
            }),
        }
    }
}

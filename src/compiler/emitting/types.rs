use std::collections::HashMap;
use super::super::lowering::lcr;

pub(super) fn resolve_type(
    s_block: &lcr::SBlock,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    var_scope: &(HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
) -> lcr::TypeRef {
    resolve_type_inner(s_block, func_map, var_scope, &mut HashMap::new(), item_map, false)
}

fn resolve_type_inner<'b>(
    s_block: &'b lcr::SBlock,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    var_scope: &(HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    temp_var_scope: &mut HashMap<&'b str, &'b lcr::TypeRef>,  // fixme merge with main var_scope
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    rec: bool,
) -> lcr::TypeRef {
    match &*s_block.tag {
        lcr::SBlockTag::Simple { code, closed, decls } => {
            for decl in decls {
                temp_var_scope.insert(
                    &decl.name,
                    &decl.r#type,
                );
            };

            // fixme if closed - doesnt mean wont never return 
            if *closed || code.is_empty() {
                lcr::TypeRef::Nothing
            } else {
                for instr in code.iter() {
                    return match instr {
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. }) if !rec =>
                            lcr::TypeRef::Never,
                        lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, .. }) =>
                            if !rec
                                && let lcr::TypeRef::Function(func) = resolve_type_inner(what, func_map, var_scope, temp_var_scope, item_map, rec)
                                && matches!(func.r#return, lcr::TypeRef::Never) {
                                lcr::TypeRef::Never
                            } else {
                                continue
                            }
                        _ => continue,
                    };
                };

                match code.last().unwrap() {
                    lcr::Instruction::LoadLiteral(lit) =>
                        lcr::TypeRef::Primitive(lit_type(lit)),
                    lcr::Instruction::Construct(lcr::ConstructInstruction::Data { what, .. }) =>
                        lcr::TypeRef::Data(what.clone()),
                    lcr::Instruction::Construct(lcr::ConstructInstruction::Array { vals, .. }) =>
                    // fixme forbid arrays of nothing
                        lcr::TypeRef::Array(vals.first().map(|b| Box::new(resolve_type_inner(b, func_map, var_scope, temp_var_scope, item_map, rec)))),
                    lcr::Instruction::DoBlock(b) => resolve_type_inner(b, func_map, var_scope, temp_var_scope, item_map, true),
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item }) => {
                        if let Some(var_name) = item.var.as_ref()
                            && let Some(r#type) = 
                                temp_var_scope.get(&**var_name).copied()
                                    .or_else(|| Some(&var_scope.0.get(var_name)?.1)) {
                            r#type.clone()
                        } else {
                            assert!(item.path.crate_.is_none());
                            item_map.get(&item.path).map(|(_, t)| lcr::TypeRef::Primitive(t.clone()))
                                .or_else(|| Some(lcr::TypeRef::Function(Box::new(func_map.get(&item.path)?.1.clone()))))
                                .unwrap_or_else(|| panic!("no such var/item/func: {item:?}\n--- scopes ---\nvar: {var_scope:?}\ntvar: {temp_var_scope:?}\nitems: {item_map:?}\nfuncs: {func_map:?})"))
                        }
                    },
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Access { .. }) => todo!("fields are not supported"),
                    lcr::Instruction::DoAction(lcr::ActionInstruction::MethodCall { .. }) => todo!("methods are not supported"),
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, .. }) =>
                        if let lcr::TypeRef::Function(func) = resolve_type_inner(what, func_map, var_scope, temp_var_scope, item_map, rec) {
                            func.r#return
                        } else {
                            panic!("cant call non functions")
                        },
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment { .. }) => lcr::TypeRef::Nothing,
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::VmDebug { .. }) => lcr::TypeRef::Nothing,
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. }) => todo!(),
                }
            }
        },
        lcr::SBlockTag::Handle { what, .. }
        | lcr::SBlockTag::Unhandle { what } =>
            resolve_type_inner(what, func_map, var_scope, temp_var_scope, item_map, rec),
        lcr::SBlockTag::Condition { code, .. } =>
            resolve_type_inner(code, func_map, var_scope, temp_var_scope, item_map, true),
        lcr::SBlockTag::Selector { cases, .. } =>
            cases.first().map(|(_, b)| resolve_type_inner(b, func_map, var_scope, temp_var_scope, item_map, true)).unwrap_or(lcr::TypeRef::Nothing),
        lcr::SBlockTag::Loop { code } =>
            if let lcr::TypeRef::Never = resolve_type_inner(code, func_map, var_scope, temp_var_scope, item_map, rec) {
                lcr::TypeRef::Never
            } else if let lcr::SBlockTag::Simple { code, .. } = &*code.tag {
                for instr in code {
                    if let lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { value: Some(value), .. }) = instr {
                        return resolve_type_inner(value, func_map, var_scope, temp_var_scope, item_map, rec)
                    };
                };
                lcr::TypeRef::Never
            } else {
                panic!("uh oh")
            },
        lcr::SBlockTag::While { .. }
        | lcr::SBlockTag::Over { .. } =>
            lcr::TypeRef::Nothing,
    }
}


/// matches resolve_type()
/// for use when no need to know precise type
/// None - !, Some(false) - nothing, Some(true) - 1 item
pub(super) fn returns_something(s_block: &lcr::SBlock, func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>) -> Option<bool> {
    returns_something_inner(s_block, func_map, false)
}


fn returns_something_inner(s_block: &lcr::SBlock, func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>, rec: bool) -> Option<bool> {
    match &*s_block.tag {
        &lcr::SBlockTag::Simple { ref code, closed, decls: _ } => {
            if closed || code.is_empty() {
                Some(false)
            } else {
                for instr in code {
                    match instr {
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. })
                        if !rec
                            => return None,
                        // xxx make it work in the future for closures
                        lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, .. })
                        if !rec
                            && let lcr::TypeRef::Function(func) = resolve_type_inner(what, func_map, &(HashMap::new(), 0), &mut HashMap::new(), &HashMap::new(), rec)
                            && matches!(func.r#return, lcr::TypeRef::Never)
                            => return None,
                        _ => {},
                    };
                };
             
                match code.last().expect("empty code has already been checked") {
                    lcr::Instruction::DoBlock(b) => returns_something_inner(b, func_map, rec),
                    lcr::Instruction::LoadLiteral(_) => Some(true),
                    lcr::Instruction::Construct(_) => Some(true),  // fixme this will work unless one of args is none
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Load { .. }) => Some(true),
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, .. }) =>
                        if let lcr::TypeRef::Function(func) = resolve_type_inner(what, func_map, &(HashMap::new(), 0), &mut HashMap::new(), &HashMap::new(), rec) {
                            match func.r#return {
                                lcr::TypeRef::Never => None,
                                lcr::TypeRef::Nothing => Some(false),
                                _ => Some(true),
                            }
                        } else {
                            panic!("cant call non-functions")
                        },
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Access { .. }) => todo!(), 
                    lcr::Instruction::DoAction(lcr::ActionInstruction::MethodCall { .. }) => todo!(),
                    lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. }) => todo!(),
                    lcr::Instruction::DoStatement(_) => Some(false),
                }
            }
        },
        lcr::SBlockTag::Handle { what, handlers: _, fallback: _ }
        | lcr::SBlockTag::Unhandle { what }
            => returns_something_inner(what, func_map, rec),
        lcr::SBlockTag::Condition { code, check: _, otherwise: _ } => returns_something_inner(code, func_map, true),
        lcr::SBlockTag::Selector { of: _, cases, fallback: _ }
            => cases.first().map_or(Some(false), |(_, b)| returns_something_inner(b, func_map, true)),
        lcr::SBlockTag::Loop { code } => todo!(),
        lcr::SBlockTag::While { check: _, code: _, do_first: _ }
        | lcr::SBlockTag::Over { what: _, with: _, code: _ } => Some(false)
    }
}


pub(super) fn lit_type(lit: &lcr::LiteralValue) -> lcr::PrimitiveType {
    match lit {
        lcr::LiteralValue::Integer(n) =>
            match n {
                lcr::LiteralInteger::I8(_) => lcr::PrimitiveType::Integer { signed: true, size: lcr::IntegerTypeSize::Byte },
                lcr::LiteralInteger::I16(_) => lcr::PrimitiveType::Integer { signed: true, size: lcr::IntegerTypeSize::Word },
                lcr::LiteralInteger::I32(_) => lcr::PrimitiveType::Integer { signed: true, size: lcr::IntegerTypeSize::DWord },
                lcr::LiteralInteger::I64(_) => lcr::PrimitiveType::Integer { signed: true, size: lcr::IntegerTypeSize::QWord },
                lcr::LiteralInteger::I128(_) => lcr::PrimitiveType::Integer { signed: true, size: lcr::IntegerTypeSize::OWord },
                lcr::LiteralInteger::U8(_) => lcr::PrimitiveType::Integer { signed: false, size: lcr::IntegerTypeSize::Byte },
                lcr::LiteralInteger::U16(_) => lcr::PrimitiveType::Integer { signed: false, size: lcr::IntegerTypeSize::Word },
                lcr::LiteralInteger::U32(_) => lcr::PrimitiveType::Integer { signed: false, size: lcr::IntegerTypeSize::DWord },
                lcr::LiteralInteger::U64(_) => lcr::PrimitiveType::Integer { signed: false, size: lcr::IntegerTypeSize::QWord },
                lcr::LiteralInteger::U128(_) => lcr::PrimitiveType::Integer { signed: false, size: lcr::IntegerTypeSize::OWord },
            },
        lcr::LiteralValue::Char(_) => lcr::PrimitiveType::Char,
        lcr::LiteralValue::String(_) => lcr::PrimitiveType::String,
        lcr::LiteralValue::Float(_) => lcr::PrimitiveType::Float,
        lcr::LiteralValue::Bool(_) => lcr::PrimitiveType::Bool,
        lcr::LiteralValue::Void => lcr::PrimitiveType::Void,
    }
}


impl lcr::TypeRef {
    pub fn within(&self, bounds: &Self) -> bool {
        match self {
            Self::Op(ops) => todo!("protos are not supported"),
            Self::Data(data) => matches!(bounds, Self::Data(other) if data == other),
            Self::Primitive(lit) => matches!(bounds, Self::Primitive(other) if lit == other),
            Self::Array(None) => true,
            Self::Array(Some(item)) => matches!(bounds, Self::Array(other) if item.within(&*other.as_ref().expect("XXX needs considiration"))),  // xxx
            Self::Function(_) => false,  // fixme
            Self::Nothing => matches!(bounds, Self::Nothing),
            Self::Never => true,
        }
    }
}

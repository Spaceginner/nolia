use std::collections::HashMap;
use super::super::lowering::lcr;

pub fn resolve_type(
    s_block: &lcr::SBlock,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    var_scope: &(HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
) -> lcr::TypeRef {
    resolve_type_inner(s_block, func_map, var_scope, &mut HashMap::new(), item_map, false)
}

// todo type resolve caching
fn resolve_type_inner<'b>(
    s_block: &'b lcr::SBlock,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    var_scope: &(HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    temp_var_scope: &mut HashMap<&'b str, &'b lcr::TypeRef>,  // fixme merge with main var_scope
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    opt_path: bool,
) -> lcr::TypeRef {
    match &*s_block.tag {
        lcr::SBlockTag::Block { block } => resolve_type_inner(block, func_map, var_scope, temp_var_scope, item_map, opt_path),
        lcr::SBlockTag::Simple { code, closed, decls } => {
            for decl in decls {
                temp_var_scope.insert(
                    &decl.name,
                    &decl.r#type,
                );
            };

            if code.is_empty() {
                lcr::TypeRef::Nothing
            } else {
                let mut can_escape_out = false;

                let mut block_type = None;
                for instr in code.iter() {
                    match instr {
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. })
                        if !opt_path
                            => block_type = Some(lcr::TypeRef::Never),
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { label, .. })
                        if matches!(s_block.label.as_ref(), Some(l) if l == label)
                            => can_escape_out = true,
                        lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, .. }) =>
                            if !opt_path
                                && let lcr::TypeRef::Function(func) = resolve_type_inner(what, func_map, var_scope, temp_var_scope, item_map, opt_path)
                                && matches!(func.r#return, lcr::TypeRef::Never) {
                                block_type = Some(lcr::TypeRef::Never)
                            }
                        _ => {},
                    };
                };

                if block_type.is_none() {
                    block_type = Some(match code.last().unwrap() {
                        lcr::Instruction::LoadLiteral(lit) =>
                            lcr::TypeRef::Primitive(lit_type(lit)),
                        lcr::Instruction::Construct(lcr::ConstructInstruction::Data { what, .. }) =>
                            lcr::TypeRef::Data(what.clone()),
                        lcr::Instruction::Construct(lcr::ConstructInstruction::Array { vals, .. }) =>
                        // fixme forbid arrays of nothing
                            lcr::TypeRef::Array(vals.first().map(|b| Box::new(resolve_type_inner(b, func_map, var_scope, temp_var_scope, item_map, opt_path)))),
                        lcr::Instruction::DoBlock(b) => resolve_type_inner(b, func_map, var_scope, temp_var_scope, item_map, false),
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
                        }
                        lcr::Instruction::DoAction(lcr::ActionInstruction::Access { .. }) => todo!("fields are not supported"),
                        lcr::Instruction::DoAction(lcr::ActionInstruction::MethodCall { .. }) => todo!("methods are not supported"),
                        lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, .. }) =>
                            if let lcr::TypeRef::Function(func) = resolve_type_inner(what, func_map, var_scope, temp_var_scope, item_map, opt_path) {
                                func.r#return
                            } else {
                                panic!("cant call non functions")
                            },
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment { .. }) => lcr::TypeRef::Nothing,
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::VmDebug { .. }) => lcr::TypeRef::Nothing,
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. })
                        | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. }) => lcr::TypeRef::Never,
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { label, value })
                        if matches!(s_block.label.as_ref(), Some(l) if l == label)
                        => value.as_ref().map_or(lcr::TypeRef::Nothing, |v| resolve_type_inner(v, func_map, var_scope, temp_var_scope, item_map, false)),
                        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { .. }) => lcr::TypeRef::Never,
                    });
                }
                
                let block_type = block_type.expect("last instr guaranteed that it is some");
                
                if matches!(block_type, lcr::TypeRef::Never) {
                    dbg!(&s_block);
                    if !(can_escape_out || dbg!(can_escape(s_block, s_block))) {
                        lcr::TypeRef::Never
                    } else {
                        lcr::TypeRef::Nothing
                    }
                } else if *closed {
                    lcr::TypeRef::Nothing
                } else {
                    block_type
                }
            }
        },
        lcr::SBlockTag::Handle { what, .. }
        | lcr::SBlockTag::Unhandle { what } =>
            resolve_type_inner(what, func_map, var_scope, temp_var_scope, item_map, opt_path),
        lcr::SBlockTag::Selector { cases, .. } =>
            cases.first().map(|(_, b)| resolve_type_inner(b, func_map, var_scope, temp_var_scope, item_map, true)).unwrap_or(lcr::TypeRef::Nothing),
    }
}


/// matches resolve_type()
/// for use when no need to know precise type
/// None - !, Some(false) - nothing, Some(true) - 1 item
pub fn returns_something(s_block: &lcr::SBlock, func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>) -> Option<bool> {
    returns_something_inner(s_block, func_map, false)
}


fn returns_something_inner(s_block: &lcr::SBlock, func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>, rec: bool) -> Option<bool> {
    match &*s_block.tag {
        lcr::SBlockTag::Block { block } => returns_something_inner(block, func_map, rec),
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
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { .. })
                    | lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. }) => unreachable!(),
                    lcr::Instruction::DoStatement(_) => Some(false),
                }
            }
        },
        lcr::SBlockTag::Handle { what, handlers: _, fallback: _ }
        | lcr::SBlockTag::Unhandle { what }
            => returns_something_inner(what, func_map, rec),
        lcr::SBlockTag::Selector { of: _, cases, fallback: _ }
            => cases.first().map_or(Some(false), |(_, b)| returns_something_inner(b, func_map, true)),
    }
}



fn can_escape(cur_s_block: &lcr::SBlock, within: &lcr::SBlock) -> bool {
    let expected_label = match cur_s_block.label.as_ref() {
        Some(l) => l,
        None => return match &*cur_s_block.tag {
            lcr::SBlockTag::Block { block } => can_escape(block, block),
            lcr::SBlockTag::Simple { code, closed: _, decls: _ }
            if code.len() == 1 => match code.first().expect("just checked for len") {
                lcr::Instruction::DoBlock(block) => can_escape(block, block),
                _ => false,
            },
            _ => false,
        },
    };

    match &*within.tag {
        lcr::SBlockTag::Block { block } => can_escape(cur_s_block, block),
        lcr::SBlockTag::Simple { code, closed: _, decls: _ }
            => code.iter().any(|instr| match instr {
                lcr::Instruction::DoBlock(block) => can_escape(cur_s_block, block),
                lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { label, value: _ }) if label == expected_label => true,
                _ => false,
            }),
        lcr::SBlockTag::Selector { of, cases, fallback}
            => can_escape(cur_s_block, of) || fallback.as_ref().is_some_and(|f| can_escape(cur_s_block, f))
                || cases.iter().any(|(check, code)| can_escape(cur_s_block, check) || can_escape(cur_s_block, code)),
        lcr::SBlockTag::Handle { what, handlers, fallback }
            => can_escape(cur_s_block, what) || fallback.as_ref().is_some_and(|(_, f)| can_escape(cur_s_block, f)) || handlers.iter().any(|(_, _, handler)| can_escape(cur_s_block, handler)),
        lcr::SBlockTag::Unhandle { what }
            => can_escape(cur_s_block, what),
    }
}


pub fn lit_type(lit: &lcr::LiteralValue) -> lcr::PrimitiveType {
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
            Self::Array(Some(item)) => matches!(bounds, Self::Array(other) if item.within(other.as_ref().expect("XXX needs consideration"))),  // xxx
            Self::Function(_) => false,  // fixme
            Self::Nothing => matches!(bounds, Self::Nothing),
            Self::Never => true,
        }
    }
}

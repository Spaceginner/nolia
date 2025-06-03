use std::collections::HashMap;
use crate::vm;
use super::super::lowering::lcr;
use super::types::resolve_type;

pub fn compile_function(
    func: lcr::Function,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &mut Vec<vm::ConstItem>,
) -> vm::FunctionDeclaration {
    vm::FunctionDeclaration {
        code: match func.code {
            lcr::Block::Structured(s_block) => compile_s_func(s_block, func.r#type, func_map, item_map, items),
            lcr::Block::Asm(asm) => compile_asm_func(asm, func_map, items),
        }
    }
}

fn compile_s_func(
    s_block: lcr::SBlock,
    func_type: lcr::FunctionType,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &mut Vec<vm::ConstItem>,
) -> Vec<vm::Op> {
    let arg_count = func_type.captures.len();
    let init_code = vec![
        vm::Op::Copy {
            count: arg_count,
            offset: 0,
        },
    ];

    let mut var_scope =
        (
            func_type.captures.into_iter().enumerate()
                .map(|(i, (name, r#type))| (name, (i, r#type)))
                .collect(),
            arg_count,
        );

    if !resolve_type(&s_block, func_map, &var_scope, item_map).within(&func_type.r#return) {
        panic!("return value is of wrong return type");
    };

    let code = compile_s_block(
        s_block,
        1,
        &func_type.r#return,
        &mut var_scope,
        &mut HashMap::new(),
        func_map,
        item_map,
        items,
    );

    let deinit_code = vec![
        vm::Op::Pop {
            count: arg_count,
            offset: if func_type.r#return.within(&lcr::TypeRef::Nothing) { 0 } else { 1 }
        },
    ];

    [
        init_code,
        code,
        deinit_code,
    ].concat()
}

fn compile_asm_func(
    asm_block: lcr::AsmBlock,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    items: &mut Vec<vm::ConstItem>,
) -> Vec<vm::Op> {
    let label_store = asm_block.code.iter().enumerate()
        .filter_map(|(i, instr)| Some((instr.label.clone()?, i)))
        .collect::<HashMap<_, _>>();

    asm_block.code.into_iter().map(|instr| match instr.op {
        lcr::AsmOp::Pack { .. } => { todo!("pack instr is not supported") }
        lcr::AsmOp::LoadConstItem { item } =>
        // todo add a way to refer to actual const items
            vm::Op::LoadConstItem {
                id: item.either(
                    |id| id.into(),
                    |val| {
                        let i = items.len();
                        items.push(val.into());
                        i.into()
                    },
                ),
            },
        lcr::AsmOp::LoadFunction { func } =>
            vm::Op::LoadFunction {
                id: func.either(
                    |id| id.into(),
                    |p| {
                        assert!(p.crate_.is_none());
                        func_map[&p].0.into()
                    },
                )
            },
        lcr::AsmOp::LoadImplementation { .. } => { todo!("load impl instr is not supported") },
        lcr::AsmOp::LoadSystemItem { id } =>
            vm::Op::LoadSystemItem {
                id: id.either(
                    |id| id.into(),
                    |name| match &name[..] {  // fixme soft-code sysitem name matching
                        "false" => vm::SysItemId::False,
                        "true" => vm::SysItemId::True,
                        "void" => vm::SysItemId::Void,
                        _ => panic!("unknown sys item")
                    },
                ),
            },
        lcr::AsmOp::Access { .. } => { todo!("access instr is not supported") }
        lcr::AsmOp::GetType => vm::Op::GetType,
        lcr::AsmOp::Call { which } => vm::Op::Call { which },
        lcr::AsmOp::SystemCall { id } =>
            vm::Op::SystemCall {
                id: id.either(
                    |id| id.into(),
                    |name| match &name[..] {  // fixme soft-code syscall name matching
                        "panic" => vm::SysCallId::Panic,
                        "println" => vm::SysCallId::PrintLine,
                        "debug" => vm::SysCallId::Debug,
                        "add" => vm::SysCallId::Add,
                        _ => panic!("unknown sys call")
                    }
                )
            },
        lcr::AsmOp::Return => vm::Op::Return,
        lcr::AsmOp::Swap { with } => vm::Op::Swap { with },
        lcr::AsmOp::Pull { which } => vm::Op::Pull { which },
        lcr::AsmOp::Pop { count, offset } => vm::Op::Pop { count, offset },
        lcr::AsmOp::Copy { count, offset } => vm::Op::Copy { count, offset },
        lcr::AsmOp::Jump { to, check } =>
            vm::Op::Jump {
                check,
                to: to.either(
                    |i| i,
                    |label| label_store[&label]
                ),
            },
    }).collect()
}


fn compile_instruction(
    instr: lcr::Instruction,
    block_starts_at: usize,
    this_instr_at: usize,
    ret_type: &lcr::TypeRef,
    var_scope: &mut (HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    label_scope: &mut HashMap<&str, usize>,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &mut Vec<vm::ConstItem>,
) -> Vec<vm::Op> {
    match instr {
        lcr::Instruction::DoBlock(block) =>
            compile_s_block(block, this_instr_at, ret_type, var_scope, label_scope, func_map, item_map, items),
        lcr::Instruction::LoadLiteral(lit) => {
            var_scope.1 += 1;
            vec![match lit {
                lcr::LiteralValue::Bool(bool) => vm::Op::LoadSystemItem { id: if bool { vm::SysItemId::True } else { vm::SysItemId::False } },
                lcr::LiteralValue::Void => vm::Op::LoadSystemItem { id: vm::SysItemId::Void },
                lit => vm::Op::LoadConstItem {
                    id: {
                        let id = items.len();
                        items.push(lit.into());
                        id.into()
                    },
                },
            }]
        },
        lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item }) => {
            if let Some(var) = item.var.as_ref()
                && let Some(&(i, _)) = var_scope.0.get(var) {
                let code = vec![vm::Op::Copy { offset: var_scope.1 - i - 1, count: 1 }];
                var_scope.1 += 1;
                code
            } else {
                var_scope.1 += 1;
                assert!(item.path.crate_.is_none());
                if let Some((i, _)) = item_map.get(&item.path) {
                    vec![vm::Op::LoadConstItem { id: (*i).into() }]
                } else if let Some((i, _)) = func_map.get(&item.path) {
                    vec![vm::Op::LoadFunction { id: (*i).into() }]
                } else {
                    panic!("such item doesnt exist");
                }
            }
        },
        lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, args }) => {
            if let lcr::TypeRef::Function(func) = resolve_type(&what, func_map, var_scope, item_map) {
                let lcr::FunctionType { generics, captures, r#return, .. } = *func;
                assert!(generics.is_empty(), "generics are not supported");

                if args.len() != captures.len() {
                    panic!("function expected different number of arguments");
                };

                if !args.iter().map(|arg| resolve_type(arg, func_map, var_scope, item_map)).zip(captures.into_iter()).all(|(arg, (_, cap))| arg.within(&cap)) {
                    panic!("some args have mismatched types");
                };

                let get_func = compile_s_block(what, this_instr_at, ret_type, var_scope, label_scope, func_map, item_map, items);
                let mut offset = get_func.len();
                let mut args_code = Vec::new();
                let arg_count = args.len();
                for arg in args {
                    let mut arg_code = compile_s_block(arg, this_instr_at + offset, ret_type, var_scope, label_scope, func_map, item_map, items);
                    offset += arg_code.len();
                    args_code.append(&mut arg_code);
                };

                let ret_count = if r#return.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };

                var_scope.1 -= arg_count + 1 - ret_count;

                [
                    get_func,
                    args_code,
                    vec![
                        vm::Op::Call { which: arg_count },
                        vm::Op::Pop {
                            count: arg_count + 1,
                            offset: ret_count,
                        },
                    ]
                ].concat()
            } else {
                panic!("cant call not functions")
            }
        },
        lcr::Instruction::DoAction(lcr::ActionInstruction::Access { of, field }) => todo!("fields access not supported"),
        lcr::Instruction::DoAction(lcr::ActionInstruction::MethodCall { what, method, args }) => todo!("methods are not supported"),
        lcr::Instruction::Construct(lcr::ConstructInstruction::Data { what, fields }) => todo!("datas are not supported"),
        lcr::Instruction::Construct(lcr::ConstructInstruction::Array { vals }) => todo!("arrays are not supported"),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment { what, to }) => {
            let exp_type = resolve_type(&what, func_map, var_scope, item_map);
            let asg_type = resolve_type(&to, func_map, var_scope, item_map);

            if !asg_type.within(&exp_type) {
                panic!("wrong type when assigning");
            };

            if let lcr::SBlockTag::Simple { code, .. } = &*what.tag {
                assert_eq!(code.len(), 1);
                match &code[0] {
                    lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item, .. }) => {
                        let replace_pos = var_scope.0[item.var.as_ref().unwrap()].0;

                        let value = compile_s_block(to, this_instr_at, ret_type, var_scope, label_scope, func_map, item_map, items);
                        var_scope.1 -= 1;

                        [
                            value,
                            vec![
                                vm::Op::Swap {
                                    with: var_scope.1 - replace_pos,
                                },
                                vm::Op::Pop {
                                    count: 1,
                                    offset: 0,
                                },
                            ]
                        ].concat()
                    },
                    _ => todo!()  // todo support fields
                }
            } else {
                panic!("cant assign to non-simple blocks")
            }
        },
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { label }) => todo!(),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { label, value }) => todo!(),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { value }) => {
            if !value.as_ref().map_or(lcr::TypeRef::Nothing, |v| resolve_type(v, func_map, var_scope, item_map)).within(ret_type) {
                panic!("function return type mismatch");
            };

            if let Some(ret_v) = value {
                let ret_code = compile_s_block(ret_v, this_instr_at, ret_type, var_scope, label_scope, func_map, item_map, items);
                let count = var_scope.1 - 1;
                var_scope.1 = 1;
                [
                    ret_code,
                    vec![
                        vm::Op::Pop {
                            count,
                            offset: 1,
                        },
                        vm::Op::Return,
                    ],
                ].concat()
            } else {
                let count = var_scope.1;
                var_scope.1 = 0;
                vec![
                    vm::Op::Pop {
                        count,
                        offset: 0,
                    },
                    vm::Op::Return,
                ]
            }
        },
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { error }) => todo!("exceptions are not supported"),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::VmDebug { dcode }) =>
            vec![
                vm::Op::LoadConstItem {
                    id: {
                        let id = items.len();
                        items.push(vm::ConstItem::Integer(vm::Integer::U32(dcode)));
                        id.into()
                    },
                },
                vm::Op::SystemCall {
                    id: vm::SysCallId::Debug,
                },
            ],
    }
}

fn compile_s_block(
    s_block: lcr::SBlock,
    starts_at: usize,
    ret_type: &lcr::TypeRef,
    var_scope: &mut (HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    label_scope: &mut HashMap<&str, usize>,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &mut Vec<vm::ConstItem>,
) -> Vec<vm::Op> {
    let self_type = resolve_type(&s_block, func_map, var_scope, item_map);

    match *s_block.tag {
        lcr::SBlockTag::Simple { code, mut decls, closed } => {
            let mut code_offset = 0;
            let init_code = decls.iter_mut()
                .map(|lcr::Declaration { name, r#type }| {
                    if var_scope.0.contains_key(&name[..]) {
                        panic!("variable shadowing is forbidden");
                    };

                    var_scope.0.insert(name.clone(), (var_scope.1, r#type.clone()));

                    code_offset += 1;
                    var_scope.1 += 1;
                    vec![vm::Op::LoadSystemItem { id: vm::SysItemId::Void }]
                })
                .reduce(|mut a, mut b| {
                    a.append(&mut b);
                    a
                })
                .unwrap_or_else(|| vec![]);

            let last_instr_i = code.len() - 1;
            let mut offset = init_code.len();
            // todo ignore code after noreturn instr
            let main_code = code.into_iter().enumerate()
                .map(|(i, instr)| {
                    let ret_something = !resolve_type(
                        &lcr::SBlock {
                            label: None,
                            tag: Box::new(lcr::SBlockTag::Simple {
                                closed: false,
                                decls: vec![],
                                code: vec![instr.clone()],
                            }),
                        },
                        func_map,
                        var_scope,
                        item_map,
                    ).within(&lcr::TypeRef::Nothing);
                    let mut code = compile_instruction(instr, starts_at, starts_at + offset, ret_type, var_scope, label_scope, func_map, item_map, items);
                    offset += code.len();
                    if ret_something && (closed || i != last_instr_i) {
                        var_scope.1 -= 1;
                        code.push(vm::Op::Pop { count: 1, offset: 0 });
                    };
                    code
                })
                .reduce(|mut a, mut b| {
                    a.append(&mut b);
                    a
                })
                .unwrap_or_else(|| vec![]);

            // todo dont add deinit if noreturn
            let deinit_code = if decls.is_empty() { vec![] } else {
                vec![vm::Op::Pop {
                    count: decls.len(),
                    offset: if self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 },
                }]
            };

            [init_code, main_code, deinit_code].concat()
        },
        lcr::SBlockTag::Condition { code, check, otherwise } => {
            if let Some(b) = otherwise.as_ref()
                && !resolve_type(b, func_map, var_scope, item_map).within(&self_type) {
                panic!("branch is not of the same return type");
            };

            if !resolve_type(&check, func_map, var_scope, item_map).within(&lcr::TypeRef::Primitive(lcr::PrimitiveType::Bool)) {
                panic!("check is not bool");
            };

            let ret_count = if self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };
            let check_code = compile_s_block(check, starts_at, ret_type, var_scope, label_scope, func_map, item_map, items);
            var_scope.1 -= 1;

            if let Some(otherwise_code) = otherwise {
                let main_pos = starts_at + check_code.len() + 3;
                let main_code = compile_s_block(code, main_pos, ret_type, var_scope, label_scope, func_map, item_map, items);
                let after_main_pos = main_pos + main_code.len() + 1;

                var_scope.1 -= ret_count;
                let otherwise = compile_s_block(otherwise_code, after_main_pos + 1, ret_type, var_scope, label_scope, func_map, item_map, items);
                let after_otherwise_pos = after_main_pos + otherwise.len() + 1;

                [
                    check_code,
                    vec![
                        vm::Op::Jump {
                            to: main_pos - 1,
                            check: Some(true),
                        },
                        vm::Op::Jump {
                            to: after_main_pos,
                            check: None,
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        },
                    ],
                    main_code,
                    vec![
                        vm::Op::Jump {
                            to: after_otherwise_pos,
                            check: None,
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        },
                    ],
                    otherwise,
                ].concat()
            } else {
                let main_pos = starts_at + check_code.len() + 2;
                let main_code = compile_s_block(code, main_pos, ret_type, var_scope, label_scope, func_map, item_map, items);
                let after_main_pos = main_pos + main_code.len() + 1 + 1;

                var_scope.1 -= ret_count;

                [
                    check_code,
                    vec![
                        // todo introduce a jump-else kind of instr
                        vm::Op::Jump {
                            to: after_main_pos,
                            check: Some(false),
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        },
                    ],
                    main_code,
                    vec![
                        vm::Op::Jump {
                            to: after_main_pos + 1,
                            check: None,
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        },
                    ]
                ].concat()
            }
        },
        lcr::SBlockTag::Selector { .. } => todo!(),
        lcr::SBlockTag::Handle { .. } => todo!(),
        lcr::SBlockTag::Unhandle { .. } => todo!(),
        lcr::SBlockTag::Loop { code } => {
            if !resolve_type(&code, func_map, var_scope, item_map).within(&lcr::TypeRef::Nothing) {
                panic!("code mustn't return anything (only through escaping)");
            };

            [
                compile_s_block(code, starts_at, ret_type, var_scope, label_scope, func_map, item_map, items),
                vec![vm::Op::Jump {
                    to: starts_at,
                    check: None
                }],
            ].concat()
        },
        // todo merge with simple loop
        lcr::SBlockTag::While { code, check, do_first } => {
            if !resolve_type(&code, func_map, var_scope, item_map).within(&lcr::TypeRef::Nothing) {
                panic!("code mustn't return anything");
            };

            if !resolve_type(&check, func_map, var_scope, item_map).within(&lcr::TypeRef::Primitive(lcr::PrimitiveType::Bool)) {
                panic!("check is not bool");
            };

            if do_first {
                let main_pos = starts_at + 2;
                let main_code = compile_s_block(code, starts_at, ret_type, var_scope, label_scope, func_map, item_map, items);
                let check_pos = main_pos + main_code.len();
                let check_code = compile_s_block(check, check_pos, ret_type, var_scope, label_scope, func_map, item_map, items);
                var_scope.1 -= 1;

                [
                    vec![
                        vm::Op::Jump {
                            to: main_pos,
                            check: None,
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        },
                    ],
                    main_code,
                    check_code,
                    vec![
                        vm::Op::Jump {
                            to: main_pos - 1,
                            check: Some(true),
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        },
                    ],
                ].concat()
            } else {
                let check_code = compile_s_block(check, starts_at, ret_type, var_scope, label_scope, func_map, item_map, items);
                var_scope.1 -= 1;
                let main_pos = starts_at + check_code.len() + 2;
                let main_code = compile_s_block(code, main_pos, ret_type, var_scope, label_scope, func_map, item_map, items);
                let loop_end_pos = main_pos + main_code.len() + 2;

                [
                    check_code,
                    vec![
                        vm::Op::Jump {
                            to: main_pos,
                            check: Some(true),
                        },
                        vm::Op::Jump {
                            to: loop_end_pos,
                            check: None,
                        },
                        vm::Op::Pop {
                            count: 1,
                            offset: 0,
                        }
                    ],
                    main_code,
                    vec![vm::Op::Jump {
                        to: starts_at,
                        check: None,
                    }],
                ].concat()
            }
        },
        lcr::SBlockTag::Over { .. } => todo!(),
    }
}

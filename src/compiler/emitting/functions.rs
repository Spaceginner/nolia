use std::collections::HashMap;
use crate::vm;
use super::super::lowering::lcr;
use super::types::{resolve_type, returns_something};

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
        &mut Vec::new(),
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
                    |name| vm::SysItemId::try_from(&*name).unwrap_or_else(|_| panic!("such sysitem doesnt exist: {name}")),
                ),
            },
        lcr::AsmOp::Access { .. } => { todo!("access instr is not supported") }
        lcr::AsmOp::Call { which } => vm::Op::Call { which },
        lcr::AsmOp::SystemCall { id } =>
            vm::Op::SystemCall {
                id: id.either(
                    |id| id.into(),
                    |name| vm::SysCallId::try_from(&*name).unwrap_or_else(|_| panic!("such syscall doesnt exist: {name}")),
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
    this_instr_at: usize,
    ret_type: &lcr::TypeRef,
    var_scope: &mut (HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    block_stack: &mut Vec<(Box<str>, BlockInfo)>,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &mut Vec<vm::ConstItem>,
) -> Vec<vm::Op> {
    match instr {
        lcr::Instruction::DoBlock(block) =>
            compile_s_block(block, this_instr_at, ret_type, var_scope, block_stack, func_map, item_map, items),
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

                let get_func = compile_s_block(what, this_instr_at, ret_type, var_scope, block_stack, func_map, item_map, items);
                let mut offset = get_func.len();
                let mut args_code = Vec::new();
                let arg_count = args.len();
                for arg in args {
                    let mut arg_code = compile_s_block(arg, this_instr_at + offset, ret_type, var_scope, block_stack, func_map, item_map, items);
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

                        let value = compile_s_block(to, this_instr_at, ret_type, var_scope, block_stack, func_map, item_map, items);
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
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { label }) => {
            let block_info = &block_stack.iter().rev().find(|&(n, _)| label == *n).unwrap().1;
            vec![
                vm::Op::Pop {
                    count: var_scope.1 - block_info.var_stack,
                    offset: 0,
                },
                vm::Op::Jump {
                    to: block_info.start,
                    check: None,
                },
            ]
        },
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { label, value }) => {
            let v_type = value.as_ref().map_or(lcr::TypeRef::Nothing, |v| resolve_type(v, func_map, var_scope, item_map));
            
            // has to be placed before the type check, because block_info will be borrowed later
            let v_code = if let Some(v_block) = value {
                compile_s_block(v_block, this_instr_at, ret_type, var_scope, block_stack, func_map, item_map, items)
            } else {
                vec![]
            };
            
            let block_info = &block_stack.iter().rev().find(|&(n, _)| label == *n).unwrap().1;
            
            if !v_type.within(&block_info.self_type) {
                panic!("escape type mismatch (expected: {:?}, got: {:?})", &block_info.self_type, v_type);
            };
            
            let ret_count = if block_info.self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };

            [
                v_code,
                vec![
                    vm::Op::Pop {
                        count: var_scope.1 - block_info.var_stack,
                        offset: ret_count,
                    },
                    vm::Op::Jump {
                        to: block_info.end,
                        check: None,
                    },
                ]
            ].concat()
        },
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { value }) => {
            if !value.as_ref().map_or(lcr::TypeRef::Nothing, |v| resolve_type(v, func_map, var_scope, item_map)).within(ret_type) {
                panic!("function return type mismatch");
            };

            if let Some(ret_v) = value {
                let ret_code = compile_s_block(ret_v, this_instr_at, ret_type, var_scope, block_stack, func_map, item_map, items);
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

#[derive(Debug, Clone)]
struct BlockInfo {
    start: usize,
    end: usize,
    var_stack: usize,
    self_type: lcr::TypeRef,  // xxx somehow turn into a ref
}

fn compile_s_block(
    s_block: lcr::SBlock,
    starts_at: usize,
    ret_type: &lcr::TypeRef,
    var_scope: &mut (HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    block_stack: &mut Vec<(Box<str>, BlockInfo)>,
    func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &mut Vec<vm::ConstItem>,
) -> Vec<vm::Op> {
    let cur_block_info = BlockInfo {
        start: starts_at,
        end: starts_at + estimate_size(&s_block, func_map),
        var_stack: var_scope.1,
        self_type: resolve_type(&s_block, func_map, var_scope, item_map),
    };

    if let Some(label) = s_block.label {
        block_stack.push((label, cur_block_info.clone()));
    };

    match *s_block.tag {
        lcr::SBlockTag::Block { block } => compile_s_block(block, starts_at, ret_type, var_scope, block_stack, func_map, item_map, items),
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
                    vm::Op::LoadSystemItem { id: vm::SysItemId::Void }
                })
                .collect::<Vec<_>>();

            let last_instr_i = code.len() - 1;
            let mut offset = init_code.len();
            // todo ignore code after noreturn instr
            let main_code = code.into_iter().enumerate()
                .flat_map(|(i, instr)| {
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
                    let mut code = compile_instruction(instr,starts_at + offset, ret_type, var_scope, block_stack, func_map, item_map, items);
                    offset += code.len();
                    if ret_something && (closed || i != last_instr_i) {
                        var_scope.1 -= 1;
                        code.push(vm::Op::Pop { count: 1, offset: 0 });
                    };
                    code
                })
                .collect();

            // todo dont add deinit if noreturn
            let deinit_code = if decls.is_empty() { vec![] } else {
                vec![vm::Op::Pop {
                    count: decls.len(),
                    offset: if cur_block_info.self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 },
                }]
            };

            var_scope.1 -= decls.len();

            for lcr::Declaration { name, .. } in &decls {
                var_scope.0.remove(name);
            };

            [init_code, main_code, deinit_code].concat()
        },
        lcr::SBlockTag::Condition { code, check, otherwise, inverted } => {
            if let Some(b) = otherwise.as_ref()
                && !resolve_type(b, func_map, var_scope, item_map).within(&cur_block_info.self_type) {
                panic!("branch is not of the same return type");
            };

            if !resolve_type(&check, func_map, var_scope, item_map).within(&lcr::TypeRef::Primitive(lcr::PrimitiveType::Bool)) {
                panic!("check is not bool");
            };

            let ret_count = if cur_block_info.self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };
            let check_code = compile_s_block(check, starts_at, ret_type, var_scope, block_stack, func_map, item_map, items);
            var_scope.1 -= 1;

            if let Some(otherwise_code) = otherwise {
                let main_pos = starts_at + check_code.len() + 3;
                let main_code = compile_s_block(code, main_pos, ret_type, var_scope, block_stack, func_map, item_map, items);
                let after_main_pos = main_pos + main_code.len() + 1;

                var_scope.1 -= ret_count;
                let otherwise = compile_s_block(otherwise_code, after_main_pos + 1, ret_type, var_scope, block_stack, func_map, item_map, items);
                let after_otherwise_pos = after_main_pos + otherwise.len() + 1;

                [
                    check_code,
                    vec![
                        vm::Op::Jump {
                            to: main_pos - 1,
                            check: Some(!inverted),
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
                let main_pos = starts_at + check_code.len() + 1;
                let main_code = compile_s_block(code, main_pos + 1, ret_type, var_scope, block_stack, func_map, item_map, items);
                let after_main_pos = main_pos + 1 + main_code.len() + 1;

                var_scope.1 -= ret_count;

                [
                    check_code,
                    vec![
                        // todo introduce a jump-else kind of instr
                        vm::Op::Jump {
                            to: after_main_pos,
                            check: Some(inverted),
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
        lcr::SBlockTag::Selector { of, cases, fallback } => {
            // of()
            // cases n:
            //      check()
            //      syscall equal
            //      jump $fallthrough false
            //      pop 1 0  // pop check
            //      code()
            //      jump $deinit
            // $fallthrough
            // pop 1 0  // pop check
            // ... other cases ...
            // fallback:
            //      fallback();
            // $deinit
            //      pop 1 1/0;  // pop of value

            let of_code = compile_s_block(of, starts_at, ret_type, var_scope, block_stack, func_map, item_map, items);
            let mut case_offset = starts_at + of_code.len();
            let deinit_pos = cur_block_info.end - 1;

            let mut cases_code = Vec::new();
            for (check, action) in cases {
                let check_size = estimate_size(&check, func_map);
                let fallthrough_pos = case_offset + check_size + 3 + estimate_size(&action, func_map) + 1;

                let mut case_code = [
                    compile_s_block(check, case_offset, ret_type, var_scope, block_stack, func_map, item_map, items),
                    vec![
                        vm::Op::SystemCall { id: vm::SysCallId::Equal },
                        vm::Op::Jump {
                            to: fallthrough_pos,
                            check: Some(false),
                        },
                        vm::Op::Pop {
                            count: 2,
                            offset: 0,
                        },
                    ],
                    compile_s_block(action, case_offset + check_size + 3, ret_type, var_scope, block_stack, func_map, item_map, items),
                    vec![
                        vm::Op::Jump {
                            to: deinit_pos,
                            check: None,
                        },
                        vm::Op::Pop {
                            count: 2,
                            offset: 0,
                        },
                    ],
                ].concat();

                case_offset += case_code.len();

                cases_code.append(&mut case_code);
            };

            let fallback_code = fallback.map_or_else(Vec::new, |f| compile_s_block(f, case_offset, ret_type, var_scope, block_stack, func_map, item_map, items));

            let ret_count = if cur_block_info.self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };
            let deinit_code = vec![vm::Op::Pop { count: 1, offset: ret_count }];

            [
                of_code,
                cases_code,
                fallback_code,
                deinit_code,
            ].concat()
        },
        lcr::SBlockTag::Handle { .. } => todo!(),
        lcr::SBlockTag::Unhandle { .. } => todo!(),
    }
}

fn estimate_size_instr(instr: &lcr::Instruction, func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>) -> usize {
    match instr {
        lcr::Instruction::DoBlock(block) => estimate_size(block, func_map),
        lcr::Instruction::LoadLiteral(_) => 1,
        lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item: _ }) => 1,
        lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, args }) => estimate_size(what, func_map) + args.iter().map(|a| estimate_size(a, func_map)).sum::<usize>() + 2,
        lcr::Instruction::DoAction(lcr::ActionInstruction::Access { .. }) => todo!(),
        lcr::Instruction::DoAction(lcr::ActionInstruction::MethodCall { .. }) => todo!(),
        lcr::Instruction::Construct(lcr::ConstructInstruction::Data { .. }) => todo!(),
        lcr::Instruction::Construct(lcr::ConstructInstruction::Array { .. }) => todo!(),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment { what: _, to }) => estimate_size(to, func_map) + 2,
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. }) => 2,
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { value, label: _ }) => 2 + value.as_ref().map_or(0, |v| estimate_size(v, func_map)),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { value }) => value.as_ref().map_or(0, |a| estimate_size(a, func_map)) + 2,
        lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. }) => todo!(),
        lcr::Instruction::DoStatement(lcr::StatementInstruction::VmDebug { .. }) => 2,
    }
}

fn estimate_size(block: &lcr::SBlock, func_map: &HashMap<lcr::Path, (usize, lcr::FunctionType)>) -> usize {
    match &*block.tag {
        lcr::SBlockTag::Block { block } => estimate_size(block, func_map),
        &lcr::SBlockTag::Simple { ref code, ref decls, closed } =>
            decls.len() + if decls.is_empty() { 0 } else { 1 } + {
                let last_i = code.len() - 1;

                code.iter().enumerate()
                    .map(|(i, c)| {
                        let ret_smth = matches!(returns_something(&lcr::SBlock {
                            label: None,
                            tag: Box::new(lcr::SBlockTag::Simple {
                                closed: false,
                                decls: vec![],
                                code: vec![c.clone()],
                            }),
                        }, func_map), Some(true));

                        estimate_size_instr(c, func_map) + if ret_smth && (closed || i != last_i) { 1 } else { 0 }
                    })
                    .sum::<usize>()
            },
        lcr::SBlockTag::Condition { code, check, otherwise, inverted: _ } =>
            estimate_size(check, func_map) + otherwise.as_ref().map_or_else(
                || estimate_size(code, func_map) + 4,
                |o| estimate_size(code, func_map) + estimate_size(o, func_map) + 5
            ),
        lcr::SBlockTag::Selector { of, cases, fallback } =>
            estimate_size(of, func_map)
                + cases.iter().map(|(check, code)| estimate_size(check, func_map) + estimate_size(code, func_map) + 5).sum::<usize>()
                + fallback.as_ref().map_or(0, |f| estimate_size(f, func_map))
                + 1,
        lcr::SBlockTag::Handle { .. } => todo!(),
        lcr::SBlockTag::Unhandle { .. } => todo!(),
    }
}

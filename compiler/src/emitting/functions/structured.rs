use std::collections::HashMap;
use super::super::functions::FunctionCompiler;
use super::super::types::{resolve_type, returns_something};
use super::lcr;

pub struct SFuncCompiler<'c, 'm> {
    comp: &'c mut FunctionCompiler<'m>,
    ret_type: lcr::TypeRef,
    arg_count: usize,
    var_scope: (HashMap<Box<str>, (usize, lcr::TypeRef)>, usize),
    block_stack: Vec<(Box<str>, BlockInfo)>,
}

#[derive(Debug, Clone)]
struct BlockInfo {
    start: usize,
    end: usize,
    var_stack: usize,
    self_type: lcr::TypeRef,  // xxx somehow turn into a ref
}

impl<'c, 'm> SFuncCompiler<'c, 'm> {
    pub fn new(comp: &'c mut FunctionCompiler<'m>, func_type: lcr::FunctionType) -> Self {
        let arg_count = func_type.captures.len();

        Self {
            comp,
            ret_type: func_type.r#return,
            arg_count,
            var_scope: (
                func_type.captures.into_iter().enumerate()
                    .map(|(i, (name, r#type))| (name, (i, r#type)))
                    .collect(),
                arg_count,
            ),
            block_stack: Vec::new(),
        }
    }

    pub fn compile(mut self, s_block: lcr::SBlock) -> Vec<vm::Op> {
        if !self.resolve_type(&s_block).within(&self.ret_type) {
            panic!("return value is of wrong return type");
        };

        let init_code = vec![
            vm::Op::Copy {
                count: self.arg_count,
                offset: 0,
            },
        ];

        let code = self.compile_s_block(s_block, 1);

        let deinit_code = vec![
            vm::Op::Pop {
                count: self.arg_count,
                offset: if self.ret_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 }
            },
        ];

        [
            init_code,
            code,
            deinit_code,
        ].concat()
    }

    // xxx should these 2 be moved into this impl itself?
    #[inline]
    fn resolve_type(&self, s_block: &lcr::SBlock) -> lcr::TypeRef {
        resolve_type(s_block, self.comp.func_map, &self.var_scope, self.comp.item_map)
    }

    #[inline]
    fn returns_something(&self, s_block: &lcr::SBlock) -> Option<bool> {
        returns_something(s_block, &self.comp.func_map)
    }

    fn compile_s_block(&mut self, s_block: lcr::SBlock, starts_at: usize) -> Vec<vm::Op> {
        let cur_block_info = BlockInfo {
            start: starts_at,
            end: starts_at + self.estimate_size(&s_block),
            var_stack: self.var_scope.1,
            self_type: self.resolve_type(&s_block),
        };

        if let Some(label) = s_block.label {
            self.block_stack.push((label, cur_block_info.clone()));
        };

        match *s_block.tag {
            lcr::SBlockTag::Block { block } => self.compile_s_block(block, starts_at),
            lcr::SBlockTag::Simple { code, mut decls, closed } => {
                if code.is_empty() {
                    return vec![];
                };

                let mut code_offset = 0;
                let init_code = decls.iter_mut()
                    .map(|lcr::Declaration { name, r#type }| {
                        if self.var_scope.0.contains_key(&name[..]) {
                            panic!("variable shadowing is forbidden");
                        };

                        self.var_scope.0.insert(name.clone(), (self.var_scope.1, r#type.clone()));

                        code_offset += 1;
                        self.var_scope.1 += 1;
                        vm::Op::LoadSystemItem { id: vm::SysItemId::Void }
                    })
                    .collect::<Vec<_>>();

                let last_instr_i = code.len() - 1;
                let mut offset = init_code.len();
                // todo ignore code after noreturn instr
                let main_code = code.into_iter().enumerate()
                    .flat_map(|(i, instr)| {
                        let ret_something = !self.resolve_type(&instr.clone().into()).within(&lcr::TypeRef::Nothing);
                        let mut code = self.compile_instruction(instr,starts_at + offset);
                        offset += code.len();
                        if ret_something && (closed || i != last_instr_i) {
                            self.var_scope.1 -= 1;
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

                self.var_scope.1 -= decls.len();

                for lcr::Declaration { name, .. } in &decls {
                    self.var_scope.0.remove(name);
                };

                [init_code, main_code, deinit_code].concat()
            },
            lcr::SBlockTag::Selector { of, cases, fallback } => {
                // todo add specific optimization for checking against bools (to just remove like 2 instrs...)

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

                let of_code = self.compile_s_block(of, starts_at);
                let mut case_offset = starts_at + of_code.len();
                let deinit_pos = cur_block_info.end - 1;

                let mut cases_code = Vec::new();
                for (check, action) in cases {
                    let check_size = self.estimate_size(&check);
                    let fallthrough_pos = case_offset + check_size + 3 + self.estimate_size(&action) + 1;

                    let check_code = self.compile_s_block(check, case_offset);
                    self.var_scope.1 -= 2;
                    let action_code = self.compile_s_block(action, case_offset + check_size + 3);

                    let mut case_code = [
                        check_code,
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
                        action_code,
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

                let fallback_code = fallback.map_or_else(Vec::new, |f| self.compile_s_block(f, case_offset));

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

    fn compile_instruction(&mut self, instr: lcr::Instruction, this_instr_at: usize) -> Vec<vm::Op> {
        match instr {
            lcr::Instruction::DoBlock(block) => self.compile_s_block(block, this_instr_at),
            lcr::Instruction::LoadLiteral(lit) => {
                self.var_scope.1 += 1;
                vec![match lit {
                    lcr::LiteralValue::Bool(bool) => vm::Op::LoadSystemItem { id: if bool { vm::SysItemId::True } else { vm::SysItemId::False } },
                    lcr::LiteralValue::Void => vm::Op::LoadSystemItem { id: vm::SysItemId::Void },
                    lit => vm::Op::LoadConstItem { id: self.comp.add_item(lit.into()) },
                }]
            },
            lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item }) => {
                if let Some(var) = item.var.as_ref()
                    && let Some(&(i, _)) = self.var_scope.0.get(var) {
                    let code = vec![vm::Op::Copy { offset: self.var_scope.1 - i - 1, count: 1 }];
                    self.var_scope.1 += 1;
                    code
                } else {
                    self.var_scope.1 += 1;
                    assert!(item.path.crate_.is_none());
                    if let Some((i, _)) = self.comp.item_map.get(&item.path) {
                        vec![vm::Op::LoadConstItem { id: (*i).into() }]
                    } else if let Some((i, _)) = self.comp.func_map.get(&item.path) {
                        vec![vm::Op::LoadFunction { id: (*i).into() }]
                    } else {
                        panic!("such item doesnt exist");
                    }
                }
            },
            lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, args }) => {
                if let lcr::TypeRef::Function(func) = self.resolve_type(&what) {
                    let lcr::FunctionType { generics, captures, r#return, .. } = *func;
                    assert!(generics.is_empty(), "generics are not supported");

                    if args.len() != captures.len() {
                        panic!("function expected different number of arguments");
                    };

                    if !args.iter().map(|arg| self.resolve_type(arg)).zip(captures.into_iter()).all(|(arg, (_, cap))| arg.within(&cap)) {
                        panic!("some args have mismatched types");
                    };

                    let get_func = self.compile_s_block(what, this_instr_at);
                    let mut offset = get_func.len();
                    let mut args_code = Vec::new();
                    let arg_count = args.len();
                    for arg in args {
                        let mut arg_code = self.compile_s_block(arg, this_instr_at + offset);
                        offset += arg_code.len();
                        args_code.append(&mut arg_code);
                    };

                    let ret_count = if r#return.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };

                    self.var_scope.1 -= arg_count + 1 - ret_count;

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
                let exp_type = self.resolve_type(&what);
                let asg_type = self.resolve_type(&to);

                if !asg_type.within(&exp_type) {
                    panic!("wrong type when assigning");
                };

                if let lcr::SBlockTag::Simple { code, .. } = &*what.tag {
                    assert_eq!(code.len(), 1);
                    match &code[0] {
                        lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item, .. }) => {
                            let replace_pos = self.var_scope.0[item.var.as_ref().unwrap()].0;

                            let value = self.compile_s_block(to, this_instr_at);
                            self.var_scope.1 -= 1;

                            [
                                value,
                                vec![
                                    vm::Op::Swap {
                                        with: self.var_scope.1 - replace_pos,
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
                let block_info = &self.block_stack.iter().rev().find(|&(n, _)| label == *n).unwrap().1;
                vec![
                    vm::Op::Pop {
                        count: self.var_scope.1 - block_info.var_stack,
                        offset: 0,
                    },
                    vm::Op::Jump {
                        to: block_info.start,
                        check: None,
                    },
                ]
            },
            lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { label, value }) => {
                let v_type = value.as_ref().map_or(lcr::TypeRef::Nothing, |v| self.resolve_type(v));

                // has to be placed before the type check, because block_info will be borrowed later
                let v_code = if let Some(v_block) = value {
                    self.compile_s_block(v_block, this_instr_at)
                } else {
                    vec![]
                };

                let block_info = &self.block_stack.iter().rev().find(|&(n, _)| label == *n).unwrap().1;

                if !v_type.within(&block_info.self_type) {
                    panic!("escape type mismatch (expected: {:?}, got: {:?})", &block_info.self_type, v_type);
                };

                let ret_count = if block_info.self_type.within(&lcr::TypeRef::Nothing) { 0 } else { 1 };

                [
                    v_code,
                    vec![
                        vm::Op::Pop {
                            count: self.var_scope.1 - block_info.var_stack,
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
                if !value.as_ref().map_or(lcr::TypeRef::Nothing, |v| self.resolve_type(v)).within(&self.ret_type) {
                    panic!("function return type mismatch");
                };

                if let Some(ret_v) = value {
                    let ret_code = self.compile_s_block(ret_v, this_instr_at);
                    let count = self.var_scope.1 - 1;
                    self.var_scope.1 = 1;
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
                    let count = self.var_scope.1;
                    self.var_scope.1 = 0;
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
                        id: self.comp.add_item(vm::ConstItem::Integer(vm::Integer::U32(dcode))),
                    },
                    vm::Op::SystemCall {
                        id: vm::SysCallId::Debug,
                    },
                ],
        }
    }

    fn estimate_size(&self, block: &lcr::SBlock) -> usize {
        match &*block.tag {
            lcr::SBlockTag::Block { block } => self.estimate_size(block),
            &lcr::SBlockTag::Simple { ref code, ref decls, closed } =>
                decls.len() + if decls.is_empty() { 0 } else { 1 } + if code.is_empty() { 0 } else {
                    let last_i = code.len() - 1;

                    code.iter().enumerate()
                        .map(|(i, c)| {
                            let ret_smth = matches!(self.returns_something(&c.clone().into()), Some(true));

                            self.estimate_size_instr(c) + if ret_smth && (closed || i != last_i) { 1 } else { 0 }
                        })
                        .sum::<usize>()
                },
            lcr::SBlockTag::Selector { of, cases, fallback } =>
                self.estimate_size(of)
                    + cases.iter().map(|(check, code)| self.estimate_size(check) + self.estimate_size(code) + 5).sum::<usize>()
                    + fallback.as_ref().map_or(0, |f| self.estimate_size(f))
                    + 1,
            lcr::SBlockTag::Handle { .. } => todo!(),
            lcr::SBlockTag::Unhandle { .. } => todo!(),
        }
    }

    fn estimate_size_instr(&self, instr: &lcr::Instruction) -> usize {
        match instr {
            lcr::Instruction::DoBlock(block) => self.estimate_size(block),
            lcr::Instruction::LoadLiteral(_) => 1,
            lcr::Instruction::DoAction(lcr::ActionInstruction::Load { item: _ }) => 1,
            lcr::Instruction::DoAction(lcr::ActionInstruction::Call { what, args }) => self.estimate_size(what) + args.iter().map(|a| self.estimate_size(a)).sum::<usize>() + 2,
            lcr::Instruction::DoAction(lcr::ActionInstruction::Access { .. }) => todo!(),
            lcr::Instruction::DoAction(lcr::ActionInstruction::MethodCall { .. }) => todo!(),
            lcr::Instruction::Construct(lcr::ConstructInstruction::Data { .. }) => todo!(),
            lcr::Instruction::Construct(lcr::ConstructInstruction::Array { .. }) => todo!(),
            lcr::Instruction::DoStatement(lcr::StatementInstruction::Assignment { what: _, to }) => self.estimate_size(to) + 2,
            lcr::Instruction::DoStatement(lcr::StatementInstruction::Repeat { .. }) => 2,
            lcr::Instruction::DoStatement(lcr::StatementInstruction::Escape { value, label: _ }) => 2 + value.as_ref().map_or(0, |v| self.estimate_size(v)),
            lcr::Instruction::DoStatement(lcr::StatementInstruction::Return { value }) => value.as_ref().map_or(0, |a| self.estimate_size(a)) + 2,
            lcr::Instruction::DoStatement(lcr::StatementInstruction::Throw { .. }) => todo!(),
            lcr::Instruction::DoStatement(lcr::StatementInstruction::VmDebug { .. }) => 2,
        }
    }
}
use std::collections::HashMap;
use crate::compiler::emitting::functions::FunctionCompiler;
use crate::compiler::lcr;
use crate::vm;

pub(super) struct AsmFuncCompiler<'c, 'm> {
    comp: &'c mut FunctionCompiler<'m>,
}

impl<'c, 'm> AsmFuncCompiler<'c, 'm> {
    pub(super) fn new(comp: &'c mut FunctionCompiler<'m>) -> Self {
        Self { comp }
    }

    pub(super) fn compile(
        self,
        asm_block: lcr::AsmBlock,
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
                        |val| self.comp.add_item(val.into()),
                    ),
                },
            lcr::AsmOp::LoadFunction { func } =>
                vm::Op::LoadFunction {
                    id: func.either(
                        |id| id.into(),
                        |p| {
                            assert!(p.crate_.is_none());
                            self.comp.func_map[&p].0.into()
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
}

mod structured;
mod asm;

use std::collections::HashMap;
use crate::vm;
use super::super::lowering::lcr;

pub(super) struct FunctionCompiler<'m> {
    func_map: &'m HashMap<lcr::Path, (usize, lcr::FunctionType)>,
    item_map: &'m HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
    items: &'m mut Vec<vm::ConstItem>,
}

impl<'m> FunctionCompiler<'m> {
    pub(super) fn new(
        func_map: &'m HashMap<lcr::Path, (usize, lcr::FunctionType)>,
        item_map: &'m HashMap<lcr::Path, (usize, lcr::PrimitiveType)>,
        items: &'m mut Vec<vm::ConstItem>
    ) -> Self {
        Self { func_map, item_map, items }
    }
    
    pub(super) fn compile(&mut self, func: lcr::Function) -> vm::FunctionDeclaration {
        vm::FunctionDeclaration {
            code: match func.code {
                lcr::Block::Structured(s_block) => structured::SFuncCompiler::new(self, func.r#type).compile(s_block),
                lcr::Block::Asm(asm_block) => asm::AsmFuncCompiler::new(self).compile(asm_block),
            }
        }
    }
    
    fn add_item(&mut self, item: vm::ConstItem) -> vm::Id {
        let i = self.items.len();
        self.items.push(item);
        i.into()
    }
}

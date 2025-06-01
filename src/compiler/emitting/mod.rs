mod functions;
mod types;

use std::collections::HashMap;
use crate::vm;
use super::Compiler;
use super::lowering::lcr;


#[derive(Debug)]
pub enum CompileError {

}

impl Compiler {
    pub fn compile(self, entry_func: Option<lcr::Path>) -> Result<(Vec<vm::CrateDeclaration>, Option<(vm::CrateId, u32)>), CompileError> {
        let mut entry_func_id = None;
        Ok((self.crates.into_iter().map(|(crate_id_raw, crate_)| {
            let (mut items, item_map) =
                crate_.item_store.into_iter().enumerate()
                    .map(|(i, (path, item))| {
                        let item_type = types::lit_type(&item);
                        (item.into(), (path, (i, item_type)))
                    })
                    .collect::<(Vec<_>, HashMap<_, _>)>();

            let func_map = crate_.function_store.iter().enumerate()
                .map(|(i, (p, f))| (p.clone(), (i, f.r#type.clone())))
                .collect::<HashMap<_, _>>();

            let funcs = crate_.function_store.into_values()
                .map(|func| functions::compile_function(func, &func_map, &item_map, &mut items))
                .collect();

            let crate_id = vm::CrateId::new(&crate_id_raw.0, crate_id_raw.1);

            if let Some(path) = entry_func.as_ref()
                && path.crate_.as_ref().unwrap() == &crate_id_raw.0 {
                entry_func_id = Some((crate_id, func_map[&path.clone().noc()].0 as u32));
            };

            vm::CrateDeclaration {
                id: crate_id,
                dependencies: Vec::new(),
                items,
                functions: funcs,
                implementations: HashMap::new(),
            }
        }).collect(), entry_func_id))
    }
}
